package Bragi

import "base:runtime"

import "core:fmt"
import "core:log"
import "core:math"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:strings"
import "core:time"

import "vendor:glfw"
import img "vendor:stb/image"
import vk "vendor:vulkan"

APP_VERSION: u32 : 1
ENGINE_VERSION: u32 : (0 << 22) | (0 << 12) | (1)

inImage, outImage, colourImage: Image

instance: vk.Instance
device: vk.Device
physicalDevice: vk.PhysicalDevice
maxImageDims: u32
computeFamily: u32
computeQueue: vk.Queue
commandPool: vk.CommandPool
commandBuffer: vk.CommandBuffer
descriptorPool: vk.DescriptorPool
descriptorSetLayout: vk.DescriptorSetLayout
descriptorSet: vk.DescriptorSet
pipelineLayout: vk.PipelineLayout
pipeline: vk.Pipeline
debugMessenger: vk.DebugUtilsMessengerEXT

totalComputeTime, totalWriteTime, imageCount: f64

logger: runtime.Logger

Image :: struct {
	image:         vk.Image,
	memory:        vk.DeviceMemory,
	view:          vk.ImageView,
	format:        vk.Format,
	sampler:       vk.Sampler,
	width, height: u32,
}

Buffer :: struct {
	buffer: vk.Buffer,
	memory: vk.DeviceMemory,
	mapped: rawptr,
}

Data :: struct {
	inputPathRoot:  string,
	outputPathRoot: string,
	shaderPathRoot: string,
	outputPath:     string,
}

// TODO: writing to disk seems to be a bottle neck. Might be worth looking into performance increases NOTE: I use an external HDD
main :: proc() {
	loggerLevel: log.Level
	when ODIN_DEBUG {
		tracker: mem.Tracking_Allocator
		mem.tracking_allocator_init(&tracker, context.allocator)
		context.allocator = mem.tracking_allocator(&tracker)

		defer {
			if len(tracker.allocation_map) > 0 {
				log.logf(
					.Warning,
					"=== %v allocations not freed: ===",
					len(tracker.allocation_map),
				)
				for _, entry in tracker.allocation_map {
					log.logf(.Warning, "- %v bytes @ %v", entry.size, entry.location)
				}
			}
			if len(tracker.bad_free_array) > 0 {
				log.logf(.Warning, "=== %v incorrect frees: ===", len(tracker.bad_free_array))
				for entry in tracker.bad_free_array {
					log.logf(.Warning, "- %p @ %v", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&tracker)
		}
		loggerLevel = .Debug
	} else {
		loggerLevel = .Error
	}
	logger = log.create_console_logger(loggerLevel)
	context.logger = logger
	defer log.destroy_console_logger(context.logger)

	data: Data = {
		inputPathRoot  = "./images/",
		outputPathRoot = "./output/",
		shaderPathRoot = "./shaders/",
	}

	expectPath := false
	pathExpected: u8 = 0
	acceptedFlags: u8 = 0
	for arg in os.args[1:] {
		if !expectPath {
			switch arg {
			case "-I":
				fallthrough
			case "-i":
				expectPath = true
				pathExpected = 1
			case "-O":
				fallthrough
			case "-o":
				expectPath = true
				pathExpected = 2
			case "-S":
				fallthrough
			case "-s":
				expectPath = true
				pathExpected = 4
			case:
				log.logf(.Error, "Unexpected argument: {}", arg)
				return
			}
			if acceptedFlags & pathExpected != 0 {
				log.logf(.Error, "Duplicate flag: {}", arg)
				return
			}
			continue
		}
		if !os.is_dir(arg) && !os.is_file(arg) {
			log.logf(.Error, "Expected file path, got: {}", arg)
			return
		}
		switch pathExpected {
		case 1:
			data.inputPathRoot = arg
		case 2:
			data.outputPathRoot = arg
			data.outputPath = arg
		case 4:
			data.shaderPathRoot = arg
		case:
			panic(fmt.tprintf("Unexpected value: pathExpected = {}", pathExpected))
		}
		expectPath = false
	}

	if !os.exists(data.inputPathRoot) {
		log.logf(.Error, "Image dir \"{}\" does not exist.", data.inputPathRoot)
		return
	}
	if !os.exists(data.shaderPathRoot) {
		log.logf(.Error, "Shader dir \"{}\" does not exist.", data.shaderPathRoot)
		return
	}
	if !os.is_dir(data.outputPathRoot) {
		log.logf(.Error, "Output must be a dir: {}", data.outputPathRoot)
		return
	}
	if !os.exists(data.outputPathRoot) && os.make_directory(data.outputPathRoot) != nil {
		log.logf(.Error, "Failed to create dir: {}", data.outputPathRoot)
		return
	}

	if !glfw.Init() {
		log.log(.Fatal, "Failed to initalize glfw.")
		return
	}
	defer glfw.Terminate()

	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))

	if createInstance() {
		return
	}
	defer vk.DestroyInstance(instance, nil)

	if pickPhysicalDevice() {
		return
	}

	if createLogicalDevice() {
		return
	}
	defer vk.DestroyDevice(device, nil)

	if createCommandBuffer() {
		return
	}
	defer vk.DestroyCommandPool(device, commandPool, nil)

	if createStorageImages() {
		return
	}
	defer cleanupStorageImages()

	when ODIN_DEBUG {
		vkSetupDebugMessenger()
		defer vk.DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nil)
	}

	if os.is_file(data.shaderPathRoot) {
		nameLength := len(data.shaderPathRoot)
		fileExtension := data.shaderPathRoot[nameLength - 4:nameLength]

		if fileExtension != ".spv" {
			log.logf(.Error, "Shader extension not recognised: \"{}\"", data.shaderPathRoot)
		} else {
			processShader(data.shaderPathRoot, &data)
		}
	} else {
		filepath.walk(data.shaderPathRoot, walkShaders, &data)
	}

	log.logf(.Info, "Total compute time: {}", totalComputeTime)
	log.logf(.Info, "Average compute time: {}", totalComputeTime / imageCount)
	log.logf(.Info, "Total write time: {}", totalWriteTime)
	log.logf(.Info, "Average write time: {}", totalWriteTime / imageCount)

	vk.DeviceWaitIdle(device)
}

processShader :: proc(shaderPath: string, data: ^Data) {
	if createPipeline(shaderPath) {
		return
	}

	if os.is_file(data^.inputPathRoot) {
		nameLength := len(data^.inputPathRoot)
		fileExtension := data^.inputPathRoot[nameLength - 4:nameLength]
		if fileExtension != ".jpg" && fileExtension != ".png" && fileExtension != "jpeg" {
			return
		}
		info, err := os.lstat(data^.inputPathRoot, context.temp_allocator)
		processImage(data^.inputPathRoot, fmt.tprintf("{}/{}", data^.outputPath, info.name))
		os.file_info_delete(info, context.temp_allocator)
	} else {
		filepath.walk(data^.inputPathRoot, walkImages, data)
	}

	cleanupPipeline()
}

walkShaders: filepath.Walk_Proc : proc(
	info: os.File_Info,
	in_err: os.Error,
	user_data: rawptr,
) -> (
	err: os.Error,
	skip_dir: bool,
) {
	if info.is_dir {
		return
	}

	user_data := (^Data)(user_data)
	nameLength := len(info.name)
	fileExtension := info.name[nameLength - 4:nameLength]

	if fileExtension != ".spv" {
		return
	}

	user_data^.outputPath = fmt.tprintf(
		"{}/{}",
		user_data^.outputPathRoot,
		info.name[:nameLength - 4],
	)

	if !os.exists(user_data^.outputPath) && os.make_directory(user_data^.outputPath) != nil {
		log.logf(.Error, "Failed to create dir: {}", user_data^.outputPath)
	}

	processShader(info.fullpath, (^Data)(user_data))
	free_all(context.temp_allocator)
	return
}

processImage :: proc(imagePath, outputPath: string) -> b8 {
	width, height: i32
	inPath := strings.clone_to_cstring(imagePath)
	pixels := img.load(inPath, &width, &height, nil, 4)
	delete(inPath)
	size := int(width * height * 4)

	stagingBuffer: vk.Buffer
	stagingBufferMemory: vk.DeviceMemory
	if createBuffer(
		size,
		{.TRANSFER_SRC, .TRANSFER_DST},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&stagingBuffer,
		&stagingBufferMemory,
	) {
		img.image_free(pixels)
		return false
	}

	data: rawptr
	vk.MapMemory(device, stagingBufferMemory, 0, vk.DeviceSize(size), {}, &data)

	mem.copy(data, pixels, size)
	img.image_free(pixels)
	vk.UnmapMemory(device, stagingBufferMemory)

	beginInfo: vk.CommandBufferBeginInfo = {
		sType            = .COMMAND_BUFFER_BEGIN_INFO,
		pNext            = nil,
		flags            = {},
		pInheritanceInfo = nil,
	}

	if vk.BeginCommandBuffer(commandBuffer, &beginInfo) != .SUCCESS {
		log.log(.Error, "Failed to start recording command buffer!")
		vk.DestroyBuffer(device, stagingBuffer, nil)
		vk.FreeMemory(device, stagingBufferMemory, nil)
		return true
	}

	transitionImageLayout(commandBuffer, outImage.image, .UNDEFINED, .GENERAL, 1)

	transitionImageLayout(commandBuffer, inImage.image, .UNDEFINED, .TRANSFER_DST_OPTIMAL, 1)
	copyBufferToImage(commandBuffer, stagingBuffer, inImage.image, u32(width), u32(height))
	transitionImageLayout(commandBuffer, inImage.image, .TRANSFER_DST_OPTIMAL, .GENERAL, 1)

	vk.CmdBindDescriptorSets(commandBuffer, .COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nil)
	vk.CmdBindPipeline(commandBuffer, .COMPUTE, pipeline)
	vk.CmdDispatch(commandBuffer, u32(width / 32) + 1, u32(height / 32) + 1, 1)

	transitionImageLayout(commandBuffer, outImage.image, .UNDEFINED, .TRANSFER_SRC_OPTIMAL, 1)
	copyImageToBuffer(commandBuffer, stagingBuffer, outImage.image, u32(width), u32(height))

	if vk.EndCommandBuffer(commandBuffer) != .SUCCESS {
		log.log(.Error, "Failed to end recording to command buffer!")
		vk.DestroyBuffer(device, stagingBuffer, nil)
		vk.FreeMemory(device, stagingBufferMemory, nil)
		return true
	}

	submitInfo: vk.SubmitInfo = {
		sType                = .SUBMIT_INFO,
		pNext                = nil,
		waitSemaphoreCount   = 0,
		pWaitSemaphores      = nil,
		pWaitDstStageMask    = nil,
		commandBufferCount   = 1,
		pCommandBuffers      = &commandBuffer,
		signalSemaphoreCount = 0,
		pSignalSemaphores    = nil,
	}

	if vk.QueueSubmit(computeQueue, 1, &submitInfo, 0) != .SUCCESS {
		log.log(.Error, "Failed to submit to queue!")
		vk.DestroyBuffer(device, stagingBuffer, nil)
		vk.FreeMemory(device, stagingBufferMemory, nil)
		return true
	}

	start := time.now()
	vk.DeviceWaitIdle(device)
	elapsed := time.since(start)
	secs := time.duration_seconds(elapsed)
	log.logf(.Info, "Process time: {}s", time.duration_seconds(elapsed))
	totalComputeTime += secs

	vk.MapMemory(device, stagingBufferMemory, 0, vk.DeviceSize(size), {}, &data)

	start = time.now()
	outPath := strings.clone_to_cstring(outputPath)
	img.write_jpg(outPath, width, height, 4, data, 100)
	delete(outPath)
	elapsed = time.since(start)
	secs = time.duration_seconds(elapsed)
	log.logf(.Info, "Write time: {}s", time.duration_seconds(elapsed))
	totalWriteTime += secs

	vk.UnmapMemory(device, stagingBufferMemory)
	vk.DestroyBuffer(device, stagingBuffer, nil)
	vk.FreeMemory(device, stagingBufferMemory, nil)
	imageCount += 1

	return false
}

walkImages: filepath.Walk_Proc : proc(
	info: os.File_Info,
	in_err: os.Error,
	user_data: rawptr,
) -> (
	err: os.Error,
	skip_dir: bool,
) {
	if info.is_dir {
		return
	}

	nameLength := len(info.name)
	fileExtension := info.name[nameLength - 4:nameLength]
	if fileExtension != ".jpg" && fileExtension != ".png" && fileExtension != "jpeg" {
		return
	}

	if processImage(
		info.fullpath,
		fmt.tprintf("{}/{}", (^Data)(user_data)^.outputPath, info.name),
	) {
		log.logf(.Error, "Failed to process image: \"{}\"", info.fullpath)
	}

	free_all(context.temp_allocator)
	return
}

createInstance :: proc() -> b8 {
	appInfo: vk.ApplicationInfo = {
		sType              = .APPLICATION_INFO,
		pNext              = nil,
		pApplicationName   = "Bragi",
		applicationVersion = APP_VERSION,
		pEngineName        = "Bragi",
		engineVersion      = ENGINE_VERSION,
		apiVersion         = vk.API_VERSION_1_3,
	}

	instanceInfo: vk.InstanceCreateInfo = {
		sType                   = .INSTANCE_CREATE_INFO,
		pNext                   = nil,
		flags                   = nil,
		pApplicationInfo        = &appInfo,
		enabledLayerCount       = 0,
		ppEnabledLayerNames     = nil,
		enabledExtensionCount   = 0,
		ppEnabledExtensionNames = nil,
	}

	when ODIN_DEBUG {
		extension: cstring = "VK_EXT_debug_utils"
		instanceInfo.enabledExtensionCount = 1
		instanceInfo.ppEnabledExtensionNames = &extension

		layer: cstring = "VK_LAYER_KHRONOS_validation"
		instanceInfo.enabledLayerCount = 1
		instanceInfo.ppEnabledLayerNames = &layer
		debugMessengerCreateInfo: vk.DebugUtilsMessengerCreateInfoEXT = {
			sType           = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			pNext           = nil,
			messageSeverity = {.ERROR, .WARNING, .INFO},
			messageType     = {.GENERAL, .PERFORMANCE, .VALIDATION},
			pfnUserCallback = vkDebugCallback,
			pUserData       = nil,
		}
		instanceInfo.pNext = &debugMessengerCreateInfo
	}

	if vk.CreateInstance(&instanceInfo, nil, &instance) != .SUCCESS {
		log.log(.Fatal, "Failed to create vulkan instance!")
		return true
	}

	vk.load_proc_addresses(instance)
	return false
}

findQueueFamily :: proc(physicalDevice: vk.PhysicalDevice) -> (u32, b32) {
	queueFamilyCount: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nil)
	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	defer delete(queueFamilies)
	vk.GetPhysicalDeviceQueueFamilyProperties(
		physicalDevice,
		&queueFamilyCount,
		raw_data(queueFamilies),
	)

	for queueFamily, index in queueFamilies {
		if .COMPUTE in queueFamily.queueFlags {
			return u32(index), false
		}
	}
	return 0, true
}

pickPhysicalDevice :: proc() -> b8 {
	scorePhysicalDevice :: proc(physicalDevice: vk.PhysicalDevice) -> (score: u32) {
		physicalDeviceProperties: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties)

		indices, err := findQueueFamily(physicalDevice)
		if err {
			return
		}

		if physicalDeviceProperties.deviceType == .DISCRETE_GPU {
			score += 1000
		}

		score += physicalDeviceProperties.limits.maxImageDimension2D
		return
	}

	deviceCount: u32
	vk.EnumeratePhysicalDevices(instance, &deviceCount, nil)

	if deviceCount == 0 {
		log.log(.Fatal, "No available devices!")
		return true
	}

	physicalDevices := make([]vk.PhysicalDevice, deviceCount)
	defer delete(physicalDevices)
	vk.EnumeratePhysicalDevices(instance, &deviceCount, raw_data(physicalDevices))

	physicalDeviceMap: map[^vk.PhysicalDevice]u32
	defer delete(physicalDeviceMap)
	for &physicalDevice in physicalDevices {
		physicalDeviceMap[&physicalDevice] = scorePhysicalDevice(physicalDevice)
	}

	bestScore: u32
	for pDevice, score in physicalDeviceMap {
		if (score > bestScore) {
			physicalDevice = (^vk.PhysicalDevice)(pDevice)^
			bestScore = score
		}
	}

	if physicalDevice == nil {
		log.log(.Fatal, "No suitable device found!")
		return true
	}

	physicalDeviceProperties: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties)
	maxImageDims = physicalDeviceProperties.limits.maxImageDimension2D
	return false
}

createLogicalDevice :: proc() -> b8 {
	computeFamily, _ = findQueueFamily(physicalDevice)

	queuePriority: f32 = 1.0
	queueCreateInfo: vk.DeviceQueueCreateInfo = {
		sType            = .DEVICE_QUEUE_CREATE_INFO,
		pNext            = nil,
		flags            = {},
		queueFamilyIndex = computeFamily,
		queueCount       = 1,
		pQueuePriorities = &queuePriority,
	}

	deviceFeatures: vk.PhysicalDeviceFeatures = {
		robustBufferAccess                      = false,
		fullDrawIndexUint32                     = false,
		imageCubeArray                          = false,
		independentBlend                        = false,
		geometryShader                          = false,
		tessellationShader                      = false,
		sampleRateShading                       = false,
		dualSrcBlend                            = false,
		logicOp                                 = false,
		multiDrawIndirect                       = false,
		drawIndirectFirstInstance               = false,
		depthClamp                              = false,
		depthBiasClamp                          = false,
		fillModeNonSolid                        = false,
		depthBounds                             = false,
		wideLines                               = false,
		largePoints                             = false,
		alphaToOne                              = false,
		multiViewport                           = false,
		samplerAnisotropy                       = false,
		textureCompressionETC2                  = false,
		textureCompressionASTC_LDR              = false,
		textureCompressionBC                    = false,
		occlusionQueryPrecise                   = false,
		pipelineStatisticsQuery                 = false,
		vertexPipelineStoresAndAtomics          = false,
		fragmentStoresAndAtomics                = false,
		shaderTessellationAndGeometryPointSize  = false,
		shaderImageGatherExtended               = false,
		shaderStorageImageExtendedFormats       = false,
		shaderStorageImageMultisample           = false,
		shaderStorageImageReadWithoutFormat     = false,
		shaderStorageImageWriteWithoutFormat    = false,
		shaderUniformBufferArrayDynamicIndexing = false,
		shaderSampledImageArrayDynamicIndexing  = false,
		shaderStorageBufferArrayDynamicIndexing = false,
		shaderStorageImageArrayDynamicIndexing  = false,
		shaderClipDistance                      = false,
		shaderCullDistance                      = false,
		shaderFloat64                           = false,
		shaderInt64                             = false,
		shaderInt16                             = false,
		shaderResourceResidency                 = false,
		shaderResourceMinLod                    = false,
		sparseBinding                           = false,
		sparseResidencyBuffer                   = false,
		sparseResidencyImage2D                  = false,
		sparseResidencyImage3D                  = false,
		sparseResidency2Samples                 = false,
		sparseResidency4Samples                 = false,
		sparseResidency8Samples                 = false,
		sparseResidency16Samples                = false,
		sparseResidencyAliased                  = false,
		variableMultisampleRate                 = false,
		inheritedQueries                        = false,
	}

	createInfo: vk.DeviceCreateInfo = {
		sType                   = .DEVICE_CREATE_INFO,
		pNext                   = nil,
		flags                   = {},
		queueCreateInfoCount    = 1,
		pQueueCreateInfos       = &queueCreateInfo,
		enabledLayerCount       = 0,
		ppEnabledLayerNames     = nil,
		enabledExtensionCount   = 0,
		ppEnabledExtensionNames = nil,
		pEnabledFeatures        = &deviceFeatures,
	}

	if vk.CreateDevice(physicalDevice, &createInfo, nil, &device) != .SUCCESS {
		log.log(.Fatal, "Failed to create device!")
		return true
	}

	vk.load_proc_addresses(device)
	vk.GetDeviceQueue(device, computeFamily, 0, &computeQueue)
	return false
}

createCommandBuffer :: proc() -> b8 {
	poolInfo: vk.CommandPoolCreateInfo = {
		sType            = .COMMAND_POOL_CREATE_INFO,
		pNext            = nil,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = computeFamily,
	}
	if vk.CreateCommandPool(device, &poolInfo, nil, &commandPool) != .SUCCESS {
		log.log(.Fatal, "Failed to create command pool!")
		return true
	}

	allocInfo: vk.CommandBufferAllocateInfo = {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		pNext              = nil,
		commandPool        = commandPool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}

	if vk.AllocateCommandBuffers(device, &allocInfo, &commandBuffer) != .SUCCESS {
		log.log(.Fatal, "Failed to allocate command buffer!")
		vk.DestroyCommandPool(device, commandPool, nil)
		return true
	}
	return false
}

createStorageImages :: proc() -> b8 {
	inImage.format = .R8G8B8A8_UNORM
	if createImage(
		&inImage,
		{},
		.D2,
		maxImageDims,
		maxImageDims,
		1,
		{._1},
		.OPTIMAL,
		{.TRANSFER_DST, .SAMPLED, .STORAGE},
		{.DEVICE_LOCAL},
		.EXCLUSIVE,
		0,
		nil,
	) {
		return true
	}

	if createImageView(&inImage, .D2, {.COLOR}, 1) {
		vk.DestroyImage(device, inImage.image, nil)
		vk.FreeMemory(device, inImage.memory, nil)
		return true
	}

	outImage.format = .R8G8B8A8_UNORM
	if createImage(
		&outImage,
		{},
		.D2,
		maxImageDims,
		maxImageDims,
		1,
		{._1},
		.OPTIMAL,
		{.TRANSFER_SRC, .SAMPLED, .STORAGE},
		{.DEVICE_LOCAL},
		.EXCLUSIVE,
		0,
		nil,
	) {
		vk.DestroyImageView(device, inImage.view, nil)
		vk.DestroyImage(device, inImage.image, nil)
		vk.FreeMemory(device, inImage.memory, nil)
		return true
	}

	if createImageView(&outImage, .D2, {.COLOR}, 1) {
		vk.DestroyImageView(device, inImage.view, nil)
		vk.DestroyImage(device, inImage.image, nil)
		vk.FreeMemory(device, inImage.memory, nil)
		vk.DestroyImage(device, outImage.image, nil)
		vk.FreeMemory(device, outImage.memory, nil)
		return true
	}

	return false
}

cleanupStorageImages :: proc() {
	vk.DestroyImageView(device, inImage.view, nil)
	vk.DestroyImage(device, inImage.image, nil)
	vk.FreeMemory(device, inImage.memory, nil)
	vk.DestroyImageView(device, outImage.view, nil)
	vk.DestroyImage(device, outImage.image, nil)
	vk.FreeMemory(device, outImage.memory, nil)
}

createPipeline :: proc(shaderPath: string) -> b8 {
	createShaderModule :: proc(filename: string) -> (shaderModule: vk.ShaderModule, err: bool) {
		loadShaderFile :: proc(filepath: string) -> (data: []byte, success: bool) {
			fileHandle, err := os.open(filepath, mode = (os.O_RDONLY | os.O_APPEND))
			if err != 0 {
				log.log(.Fatal, "Shader file couldn't be opened!")
				return nil, false
			}
			data, success = os.read_entire_file_from_handle(fileHandle)
			if !success {
				log.log(.Fatal, "Shader file couldn't be read!")
			}
			os.close(fileHandle)
			return
		}

		code, success := loadShaderFile(filename)
		if !success {
			return 0, true
		}
		defer delete(code)

		createInfo: vk.ShaderModuleCreateInfo = {
			sType    = .SHADER_MODULE_CREATE_INFO,
			pNext    = nil,
			flags    = {},
			codeSize = len(code),
			pCode    = (^u32)(raw_data(code)),
		}
		if vk.CreateShaderModule(device, &createInfo, nil, &shaderModule) != .SUCCESS {
			log.log(.Fatal, "Failed to create shader module!")
			return 0, true
		}
		return shaderModule, false
	}

	poolSizes: []vk.DescriptorPoolSize = {{type = .STORAGE_IMAGE, descriptorCount = 2}}
	poolInfo: vk.DescriptorPoolCreateInfo = {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		pNext         = nil,
		flags         = {},
		maxSets       = 1,
		poolSizeCount = u32(len(poolSizes)),
		pPoolSizes    = raw_data(poolSizes),
	}

	if vk.CreateDescriptorPool(device, &poolInfo, nil, &descriptorPool) != .SUCCESS {
		log.log(.Fatal, "Failed to create descriptor pool!")
		return true
	}

	layoutBindings: []vk.DescriptorSetLayoutBinding = {
		{
			binding = 0,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = 1,
			stageFlags = {.COMPUTE},
			pImmutableSamplers = nil,
		},
		{
			binding = 1,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = 1,
			stageFlags = {.COMPUTE},
			pImmutableSamplers = nil,
		},
	}

	layoutInfo: vk.DescriptorSetLayoutCreateInfo = {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = nil,
		flags        = {},
		bindingCount = u32(len(layoutBindings)),
		pBindings    = raw_data(layoutBindings),
	}

	if vk.CreateDescriptorSetLayout(device, &layoutInfo, nil, &descriptorSetLayout) != .SUCCESS {
		log.log(.Fatal, "Failed to create descriptor set layout!")
		vk.DestroyDescriptorPool(device, descriptorPool, nil)
		return true
	}

	layout := descriptorSetLayout

	allocInfo: vk.DescriptorSetAllocateInfo = {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		pNext              = nil,
		descriptorPool     = descriptorPool,
		descriptorSetCount = 1,
		pSetLayouts        = &layout,
	}

	if vk.AllocateDescriptorSets(device, &allocInfo, &descriptorSet) != .SUCCESS {
		log.log(.Fatal, "Failed to allocate descriptor sets!")
		vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)
		vk.DestroyDescriptorPool(device, descriptorPool, nil)
		return true
	}

	inImageInfo: vk.DescriptorImageInfo = {
		sampler     = inImage.sampler,
		imageView   = inImage.view,
		imageLayout = .GENERAL,
	}
	outImageInfo: vk.DescriptorImageInfo = {
		sampler     = outImage.sampler,
		imageView   = outImage.view,
		imageLayout = .GENERAL,
	}

	descriptorWrite: []vk.WriteDescriptorSet = {
		{
			sType = .WRITE_DESCRIPTOR_SET,
			pNext = nil,
			dstSet = descriptorSet,
			dstBinding = 0,
			dstArrayElement = 0,
			descriptorCount = 1,
			descriptorType = .STORAGE_IMAGE,
			pImageInfo = &inImageInfo,
			pBufferInfo = nil,
			pTexelBufferView = nil,
		},
		{
			sType = .WRITE_DESCRIPTOR_SET,
			pNext = nil,
			dstSet = descriptorSet,
			dstBinding = 1,
			dstArrayElement = 0,
			descriptorCount = 1,
			descriptorType = .STORAGE_IMAGE,
			pImageInfo = &outImageInfo,
			pBufferInfo = nil,
			pTexelBufferView = nil,
		},
	}

	vk.UpdateDescriptorSets(device, u32(len(descriptorWrite)), raw_data(descriptorWrite), 0, nil)

	PipelineLayoutInfo: vk.PipelineLayoutCreateInfo = {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		pNext                  = nil,
		flags                  = {},
		setLayoutCount         = 1,
		pSetLayouts            = &descriptorSetLayout,
		pushConstantRangeCount = 0,
		pPushConstantRanges    = nil,
	}

	if vk.CreatePipelineLayout(device, &PipelineLayoutInfo, nil, &pipelineLayout) != .SUCCESS {
		log.log(.Fatal, "Failed to create pipeline layout!")
		vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)
		vk.DestroyDescriptorPool(device, descriptorPool, nil)
		return true
	}

	module, err := createShaderModule(shaderPath)
	if err {
		vk.DestroyPipelineLayout(device, pipelineLayout, nil)
		vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)
		vk.DestroyDescriptorPool(device, descriptorPool, nil)
		return true
	}
	shaderStageInfo: vk.PipelineShaderStageCreateInfo = {
		sType               = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		pNext               = nil,
		flags               = {},
		stage               = {.COMPUTE},
		module              = module,
		pName               = "main",
		pSpecializationInfo = nil,
	}

	pipelineInfo: vk.ComputePipelineCreateInfo = {
		sType              = .COMPUTE_PIPELINE_CREATE_INFO,
		pNext              = nil,
		flags              = {},
		stage              = shaderStageInfo,
		layout             = pipelineLayout,
		basePipelineHandle = {},
		basePipelineIndex  = 0,
	}

	if vk.CreateComputePipelines(device, 0, 1, &pipelineInfo, nil, &pipeline) != .SUCCESS {
		log.log(.Fatal, "Failed to create pipeline!")
		vk.DestroyPipelineLayout(device, pipelineLayout, nil)
		vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)
		vk.DestroyDescriptorPool(device, descriptorPool, nil)
		return true
	}

	vk.DestroyShaderModule(device, shaderStageInfo.module, nil)
	return false
}

cleanupPipeline :: proc() {
	vk.DestroyPipeline(device, pipeline, nil)
	vk.DestroyPipelineLayout(device, pipelineLayout, nil)
	vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)
	vk.DestroyDescriptorPool(device, descriptorPool, nil)
}

findMemoryType :: proc(typeFilter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	memProperties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties)
	for i in 0 ..< memProperties.memoryTypeCount {
		if typeFilter & (1 << i) != 0 &&
		   (memProperties.memoryTypes[i].propertyFlags & properties) == properties {
			return i
		}
	}
	log.log(.Fatal, "Failed to find suitable memory type!")
	return ~u32(0)
}

createBuffer :: proc(
	size: int,
	usage: vk.BufferUsageFlags,
	properties: vk.MemoryPropertyFlags,
	buffer: ^vk.Buffer,
	bufferMemory: ^vk.DeviceMemory,
) -> b8 {
	bufferInfo: vk.BufferCreateInfo = {
		sType                 = .BUFFER_CREATE_INFO,
		pNext                 = nil,
		flags                 = {},
		size                  = vk.DeviceSize(size),
		usage                 = usage,
		sharingMode           = .EXCLUSIVE,
		queueFamilyIndexCount = 0,
		pQueueFamilyIndices   = nil,
	}
	if vk.CreateBuffer(device, &bufferInfo, nil, buffer) != .SUCCESS {
		log.log(.Fatal, "Failed to create buffer!")
		return true
	}

	memRequirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, buffer^, &memRequirements)
	memType := findMemoryType(memRequirements.memoryTypeBits, properties)
	if ~memType == 0 {
		log.log(.Fatal, "Failed to find suitable memory type!")
		vk.DestroyBuffer(device, buffer^, nil)
		return true
	}
	allocInfo: vk.MemoryAllocateInfo = {
		sType           = .MEMORY_ALLOCATE_INFO,
		pNext           = nil,
		allocationSize  = memRequirements.size,
		memoryTypeIndex = memType,
	}
	if vk.AllocateMemory(device, &allocInfo, nil, bufferMemory) != .SUCCESS {
		log.log(.Fatal, "Failed to allocate buffer memory!")
		vk.DestroyBuffer(device, buffer^, nil)
		return true
	}
	vk.BindBufferMemory(device, buffer^, bufferMemory^, 0)
	return false
}

createImage :: proc(
	image: ^Image,
	flags: vk.ImageCreateFlags,
	imageType: vk.ImageType,
	width, height, arrayLayers: u32,
	sampleCount: vk.SampleCountFlags,
	tiling: vk.ImageTiling,
	usage: vk.ImageUsageFlags,
	properties: vk.MemoryPropertyFlags,
	sharingMode: vk.SharingMode,
	queueFamilyIndexCount: u32,
	queueFamilyIndices: [^]u32,
) -> b8 {
	imageInfo: vk.ImageCreateInfo = {
		sType                 = .IMAGE_CREATE_INFO,
		pNext                 = nil,
		flags                 = flags,
		imageType             = imageType,
		format                = image.format,
		extent                = {width, height, 1},
		mipLevels             = 1,
		arrayLayers           = arrayLayers,
		samples               = sampleCount,
		tiling                = tiling,
		usage                 = usage,
		sharingMode           = sharingMode,
		queueFamilyIndexCount = queueFamilyIndexCount,
		pQueueFamilyIndices   = queueFamilyIndices,
		initialLayout         = .UNDEFINED,
	}

	if vk.CreateImage(device, &imageInfo, nil, &image^.image) != .SUCCESS {
		log.log(.Fatal, "Faile to create image!")
		return true
	}

	memRequirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, image.image, &memRequirements)
	memType := findMemoryType(memRequirements.memoryTypeBits, properties)
	if ~memType == 0 {
		log.log(.Fatal, "Failed to find suitable memory type!")
		vk.DestroyImage(device, image^.image, nil)
		return true
	}
	allocInfo: vk.MemoryAllocateInfo = {
		sType           = .MEMORY_ALLOCATE_INFO,
		pNext           = nil,
		allocationSize  = memRequirements.size,
		memoryTypeIndex = memType,
	}

	if vk.AllocateMemory(device, &allocInfo, nil, &image^.memory) != .SUCCESS {
		log.log(.Fatal, "Failed to allocate image memory!")
		vk.DestroyImage(device, image^.image, nil)
		return true
	}

	if vk.BindImageMemory(device, image^.image, image^.memory, 0) != .SUCCESS {
		log.log(.Fatal, "Failed to bind image memory!")
		vk.DestroyImage(device, image^.image, nil)
		vk.FreeMemory(device, image^.memory, nil)
		return true
	}
	return false
}

createImageView :: proc(
	image: ^Image,
	viewType: vk.ImageViewType,
	aspectFlags: vk.ImageAspectFlags,
	layerCount: u32,
) -> b8 {
	viewInfo: vk.ImageViewCreateInfo = {
		sType = .IMAGE_VIEW_CREATE_INFO,
		pNext = nil,
		flags = {},
		image = image^.image,
		viewType = viewType,
		format = image^.format,
		components = {r = .IDENTITY, g = .IDENTITY, b = .IDENTITY, a = .IDENTITY},
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = aspectFlags,
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = layerCount,
		},
	}

	if vk.CreateImageView(device, &viewInfo, nil, &image^.view) != .SUCCESS {
		log.log(.Fatal, "Failed to create image view!")
		return true
	}
	return false
}

transitionImageLayout :: proc(
	commandBuffer: vk.CommandBuffer,
	image: vk.Image,
	oldLayout, newLayout: vk.ImageLayout,
	layerCount: u32,
) {
	barrier: vk.ImageMemoryBarrier = {
		sType = .IMAGE_MEMORY_BARRIER,
		pNext = nil,
		srcAccessMask = {},
		dstAccessMask = {},
		oldLayout = oldLayout,
		newLayout = newLayout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = layerCount,
		},
	}
	sourceStage, destinationStage: vk.PipelineStageFlags
	#partial switch oldLayout {
	case .UNDEFINED:
		barrier.srcAccessMask = {}
		sourceStage = {.TOP_OF_PIPE}
	case .TRANSFER_SRC_OPTIMAL:
		barrier.srcAccessMask = {.TRANSFER_READ}
		sourceStage = {.TRANSFER}
	case .TRANSFER_DST_OPTIMAL:
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		sourceStage = {.TRANSFER}
	case:
		panic("Unsupported image layout transition!")
	}

	#partial switch newLayout {
	case .TRANSFER_SRC_OPTIMAL:
		barrier.dstAccessMask = {.TRANSFER_READ}
		destinationStage = {.TRANSFER}
	case .TRANSFER_DST_OPTIMAL:
		barrier.dstAccessMask = {.TRANSFER_WRITE}
		destinationStage = {.TRANSFER}
	case .SHADER_READ_ONLY_OPTIMAL:
		if (barrier.srcAccessMask == {}) {
			barrier.srcAccessMask = {.HOST_WRITE, .TRANSFER_WRITE}
		}
		barrier.dstAccessMask = {.SHADER_READ}
		destinationStage = {.FRAGMENT_SHADER}
	case .GENERAL:
		if oldLayout == .TRANSFER_SRC_OPTIMAL {
			barrier.dstAccessMask = {.SHADER_WRITE}
		} else if oldLayout == .TRANSFER_DST_OPTIMAL {
			barrier.dstAccessMask = {.SHADER_READ}
		}
		destinationStage = {.COMPUTE_SHADER}
	case .PRESENT_SRC_KHR:
		barrier.dstAccessMask = {.SHADER_READ}
		destinationStage = {.COMPUTE_SHADER}
	case:
		panic("Unsupported image layout transition!")
	}
	vk.CmdPipelineBarrier(
		commandBuffer,
		sourceStage,
		destinationStage,
		{},
		0,
		nil,
		0,
		nil,
		1,
		&barrier,
	)
}

copyBufferToImage :: proc(
	commandBuffer: vk.CommandBuffer,
	buffer: vk.Buffer,
	image: vk.Image,
	width, height: u32,
) {
	region: vk.BufferImageCopy = {
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,
		imageSubresource = vk.ImageSubresourceLayers {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		imageOffset = vk.Offset3D{x = 0, y = 0, z = 0},
		imageExtent = vk.Extent3D{width = width, height = height, depth = 1},
	}
	vk.CmdCopyBufferToImage(commandBuffer, buffer, image, .TRANSFER_DST_OPTIMAL, 1, &region)
}

copyImageToBuffer :: proc(
	commandBuffer: vk.CommandBuffer,
	buffer: vk.Buffer,
	image: vk.Image,
	width, height: u32,
) {
	region: vk.BufferImageCopy = {
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,
		imageSubresource = vk.ImageSubresourceLayers {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		imageOffset = vk.Offset3D{x = 0, y = 0, z = 0},
		imageExtent = vk.Extent3D{width = width, height = height, depth = 1},
	}
	vk.CmdCopyImageToBuffer(commandBuffer, image, .TRANSFER_SRC_OPTIMAL, buffer, 1, &region)
}

copyImage :: proc(
	commandBuffer: vk.CommandBuffer,
	extent: vk.Extent3D,
	srcImage, dstImage: vk.Image,
	srcLayout, dstLayout: vk.ImageLayout,
) {
	region: vk.ImageCopy = {
		srcSubresource = {aspectMask = {.COLOR}, mipLevel = 0, baseArrayLayer = 0, layerCount = 1},
		srcOffset = {x = 0, y = 0, z = 0},
		dstSubresource = {aspectMask = {.COLOR}, mipLevel = 0, baseArrayLayer = 0, layerCount = 1},
		dstOffset = {x = 0, y = 0, z = 0},
		extent = extent,
	}
	vk.CmdCopyImage(commandBuffer, srcImage, srcLayout, dstImage, dstLayout, 1, &region)
}

when ODIN_DEBUG {
	when ODIN_OS == .Windows {
		vkDebugCallback :: proc "std" (
			messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
			messageType: vk.DebugUtilsMessageTypeFlagsEXT,
			pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
			pUserData: rawptr,
		) -> b32 {
			context = runtime.default_context()
			context.logger = logger
			severity: string
			if .GENERAL in messageType {
				severity = "General"
			} else if .VALIDATION in messageType {
				severity = "Validation"
			} else if .PERFORMANCE in messageType {
				severity = "Performance"
			} else {
				severity = "Unknown"
			}
			log.logf(
				.Debug,
				"Vulkan validation layer ({}):\n{}\n",
				severity,
				pCallbackData.pMessage,
			)
			return false
		}
	} else {
		vkDebugCallback :: proc "cdecl" (
			messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
			messageType: vk.DebugUtilsMessageTypeFlagsEXT,
			pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
			pUserData: rawptr,
		) -> b32 {
			context = runtime.default_context()
			context.logger = logger
			severity: string
			if .GENERAL in messageType {
				severity = "General"
			} else if .VALIDATION in messageType {
				severity = "Validation"
			} else if .PERFORMANCE in messageType {
				severity = "Performance"
			} else {
				severity = "Unknown"
			}
			log.logf(
				.Debug,
				"Vulkan validation layer ({}):\n{}\n",
				severity,
				pCallbackData.pMessage,
			)
			return false
		}
	}

	vkSetupDebugMessenger :: proc() {
		createInfo: vk.DebugUtilsMessengerCreateInfoEXT = {
			sType           = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			pNext           = nil,
			messageSeverity = {.ERROR, .WARNING, .INFO},
			messageType     = {.GENERAL, .PERFORMANCE, .VALIDATION},
			pfnUserCallback = vkDebugCallback,
			pUserData       = nil,
		}
		vk.CreateDebugUtilsMessengerEXT(instance, &createInfo, nil, &debugMessenger)
	}
}
