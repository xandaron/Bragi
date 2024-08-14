package Bragi

import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:strings"
import "core:time"
import "core:math"

import "vendor:glfw"
import img "vendor:stb/image"
import vk "vendor:vulkan"

// ###################################################################
// #                          Constants                              #
// ###################################################################

APP_VERSION: u32 : 1
ENGINE_VERSION: u32 : (0 << 22) | (0 << 12) | (1)

IMAGE_PATHS :: [?]string{"./assets/textures/blue.jpg"}

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

totalComputeTime, totalWriteTime: f64
imageCount: f64

// ###################################################################
// #                       Data Structures                           #
// ###################################################################

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

// ###################################################################
// #                          Functions                              #
// ###################################################################

Data :: struct {
	outputPath: string,
	shaderPath: string,
}

main :: proc() {
	if !glfw.Init() {
		panic("Failed to initalize glfw.")
	}
	defer glfw.Terminate()

	initVkGraphics()
	defer cleanupVkGraphics()

	createStorageImages(maxImageDims, maxImageDims)
	defer cleanupStorageImages()

	// TODO: add arg to pass a specific shader
	// TODO: add arg to pass images
	// TODO: when no args are passed should compile all shaders and use on all images
	// TODO: writing to disk seems to be a bottle neck. Might be worth looking into performance increases NOTE: I use an external HDD
	data: Data = {
		outputPath = "./output/compute",
		shaderPath = "./shaders/compute.spv",
	}

	createPipeline(data.shaderPath)
	defer cleanupPipeline()

	if !os.exists(data.outputPath) {
		if os.make_directory(data.outputPath) != nil {
			panic(fmt.aprintf("Failed to create dir: {}", data.outputPath))
		}
	}
	filepath.walk("./images/", walkFunc, &data)

	fmt.printfln("Total compute time: {}", totalComputeTime)
	fmt.printfln("Average compute time: {}", totalComputeTime / imageCount)
	fmt.printfln("Total write time: {}", totalWriteTime)
	fmt.printfln("Average write time: {}", totalWriteTime / imageCount)
}

processImage :: proc(imagePath, outputPath, shaderPath: string) {
	width, height: i32
	pixels := img.load(strings.clone_to_cstring(imagePath), &width, &height, nil, 4)
	size := int(width * height * 4)

	stagingBuffer: vk.Buffer
	stagingBufferMemory: vk.DeviceMemory
	createBuffer(
		size,
		{.TRANSFER_SRC, .TRANSFER_DST},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&stagingBuffer,
		&stagingBufferMemory,
	)

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
		panic("Failed to start recording compute commands!")
	}

	transitionImageLayout(
		commandBuffer,
		inImage.image,
		.R8G8B8A8_SRGB,
		.UNDEFINED,
		.TRANSFER_DST_OPTIMAL,
		1,
	)

	copyBufferToImage(commandBuffer, stagingBuffer, inImage.image, u32(width), u32(height))

	transitionImageLayout(
		commandBuffer,
		inImage.image,
		.R8G8B8A8_SRGB,
		.TRANSFER_DST_OPTIMAL,
		.SHADER_READ_ONLY_OPTIMAL,
		1,
	)

	vk.CmdBindDescriptorSets(commandBuffer, .COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nil)

	vk.CmdBindPipeline(commandBuffer, .COMPUTE, pipeline)

	vk.CmdDispatch(commandBuffer, u32(width / 32) + 1, u32(height / 32) + 1, 1)

	transitionImageLayout(
		commandBuffer,
		outImage.image,
		outImage.format,
		.UNDEFINED,
		.TRANSFER_SRC_OPTIMAL,
		1,
	)

	copyImageToBuffer(commandBuffer, stagingBuffer, outImage.image, u32(width), u32(height))

	if vk.EndCommandBuffer(commandBuffer) != .SUCCESS {
		panic("Failed to record compute command buffer!")
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

	start := time.now()
	if vk.QueueSubmit(computeQueue, 1, &submitInfo, 0) != .SUCCESS {
		panic("Failed to submit compute command buffer!")
	}
	vk.QueueWaitIdle(computeQueue)
	elapsed := time.since(start)
	secs := time.duration_seconds(elapsed)
	fmt.printfln("Process time: {}s", time.duration_seconds(elapsed))
	totalComputeTime += secs

	vk.MapMemory(device, stagingBufferMemory, 0, vk.DeviceSize(size), {}, &data)

	start = time.now()
	img.write_jpg(strings.clone_to_cstring(outputPath), width, height, 4, data, 100)
	elapsed = time.since(start)
	secs = time.duration_seconds(elapsed)
	fmt.printfln("Write time: {}s", time.duration_seconds(elapsed))
	totalWriteTime += secs

	vk.UnmapMemory(device, stagingBufferMemory)
	vk.DestroyBuffer(device, stagingBuffer, nil)
	vk.FreeMemory(device, stagingBufferMemory, nil)
	imageCount += 1
}

walkFunc: filepath.Walk_Proc : proc(
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
	processImage(
		info.fullpath,
		fmt.aprintf("{}/{}", (^Data)(user_data)^.outputPath, info.name),
		(^Data)(user_data)^.shaderPath,
	)
	return
}

initVkGraphics :: proc() {
	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))
	createInstance()
	pickPhysicalDevice()
	createLogicalDevice()
	createCommandBuffer()
}


// ####################################################################
// #                               Init                               #
// ####################################################################


createInstance :: proc() {
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

	if vk.CreateInstance(&instanceInfo, nil, &instance) != .SUCCESS {
		panic("Failed to create vulkan instance.")
	}

	// load_proc_addresses_instance :: proc(instance: Instance)
	vk.load_proc_addresses(instance)
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

pickPhysicalDevice :: proc() {
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
		panic("No devices with Vulkan support!")
	}

	physicalDevices := make([]vk.PhysicalDevice, deviceCount)
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
		panic("No suitable physical device found!")
	}

	physicalDeviceProperties: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties)
	maxImageDims = physicalDeviceProperties.limits.maxImageDimension2D
}

createLogicalDevice :: proc() {
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
		panic("Failed to create logical device!")
	}

	// load_proc_addresses_device :: proc(device: Device)
	vk.load_proc_addresses(device)

	vk.GetDeviceQueue(device, computeFamily, 0, &computeQueue)
}

createCommandBuffer :: proc() {
	poolInfo: vk.CommandPoolCreateInfo = {
		sType            = .COMMAND_POOL_CREATE_INFO,
		pNext            = nil,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = computeFamily,
	}
	if vk.CreateCommandPool(device, &poolInfo, nil, &commandPool) != .SUCCESS {
		panic("Failed to create command pool!")
	}

	allocInfo: vk.CommandBufferAllocateInfo = {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		pNext              = nil,
		commandPool        = commandPool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	if vk.AllocateCommandBuffers(device, &allocInfo, &commandBuffer) != .SUCCESS {
		panic("Failed to allocate command buffer!")
	}
}

createBuffer :: proc(
	size: int,
	usage: vk.BufferUsageFlags,
	properties: vk.MemoryPropertyFlags,
	buffer: ^vk.Buffer,
	bufferMemory: ^vk.DeviceMemory,
) {
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
		panic("Failed to create buffer!")
	}

	memRequirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, buffer^, &memRequirements)
	allocInfo: vk.MemoryAllocateInfo = {
		sType           = .MEMORY_ALLOCATE_INFO,
		pNext           = nil,
		allocationSize  = memRequirements.size,
		memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
	}
	if vk.AllocateMemory(device, &allocInfo, nil, bufferMemory) != .SUCCESS {
		panic("Failed to allocate buffer memory!")
	}
	vk.BindBufferMemory(device, buffer^, bufferMemory^, 0)
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
	panic("Failed to find suitable memory type!")
}


// ####################################################################
// #                              Images                              #
// ####################################################################


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
) {
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
		panic("Failed to create texture!")
	}

	memRequirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, image^.image, &memRequirements)
	allocInfo: vk.MemoryAllocateInfo = {
		sType           = .MEMORY_ALLOCATE_INFO,
		pNext           = nil,
		allocationSize  = memRequirements.size,
		memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
	}
	if vk.AllocateMemory(device, &allocInfo, nil, &image.memory) != .SUCCESS {
		panic("Failed to allocate image memory!")
	}
	if vk.BindImageMemory(device, image.image, image.memory, 0) != .SUCCESS {
		panic("Failed to bind image memory!")
	}
}

createImageView :: proc(
	image: vk.Image,
	viewType: vk.ImageViewType,
	format: vk.Format,
	aspectFlags: vk.ImageAspectFlags,
	layerCount: u32,
) -> (
	imageView: vk.ImageView,
) {
	viewInfo: vk.ImageViewCreateInfo = {
		sType = .IMAGE_VIEW_CREATE_INFO,
		pNext = nil,
		flags = {},
		image = image,
		viewType = viewType,
		format = format,
		components = {r = .IDENTITY, g = .IDENTITY, b = .IDENTITY, a = .IDENTITY},
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = aspectFlags,
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = layerCount,
		},
	}
	if vk.CreateImageView(device, &viewInfo, nil, &imageView) != .SUCCESS {
		panic("Failed to create image view!")
	}
	return imageView
}

transitionImageLayout :: proc(
	commandBuffer: vk.CommandBuffer,
	image: vk.Image,
	format: vk.Format,
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

createStorageImages :: proc(width, height: u32) {
	inImage.format = .R8G8B8A8_SRGB
	createImage(
		&inImage,
		{},
		.D2,
		width,
		height,
		1,
		{._1},
		.OPTIMAL,
		{.TRANSFER_DST, .SAMPLED, .STORAGE},
		{.DEVICE_LOCAL},
		.EXCLUSIVE,
		0,
		nil,
	)

	inImage.view = createImageView(inImage.image, .D2, inImage.format, {.COLOR}, 1)

	outImage.format = .R8G8B8A8_SRGB
	createImage(
		&outImage,
		{},
		.D2,
		width,
		height,
		1,
		{._1},
		.OPTIMAL,
		{.TRANSFER_SRC, .SAMPLED, .STORAGE},
		{.DEVICE_LOCAL},
		.EXCLUSIVE,
		0,
		nil,
	)

	outImage.view = createImageView(outImage.image, .D2, outImage.format, {.COLOR}, 1)
}

cleanupStorageImages :: proc() {
	vk.DestroyImageView(device, inImage.view, nil)
	vk.DestroyImage(device, inImage.image, nil)
	vk.FreeMemory(device, inImage.memory, nil)
	vk.DestroyImageView(device, outImage.view, nil)
	vk.DestroyImage(device, outImage.image, nil)
	vk.FreeMemory(device, outImage.memory, nil)
}

createPipeline :: proc(shaderPath: string) {
	createShaderModule :: proc(filename: string) -> (shaderModule: vk.ShaderModule) {
		loadShaderFile :: proc(filepath: string) -> (data: []byte) {
			fileHandle, err := os.open(filepath, mode = (os.O_RDONLY | os.O_APPEND))
			if err != 0 {
				panic("Shader file couldn't be opened!")
			}
			defer os.close(fileHandle)
			success: bool
			if data, success = os.read_entire_file_from_handle(fileHandle); !success {
				panic("Shader file couldn't be read!")
			}
			return
		}

		code := loadShaderFile(filename)
		createInfo: vk.ShaderModuleCreateInfo = {
			sType    = .SHADER_MODULE_CREATE_INFO,
			pNext    = nil,
			flags    = {},
			codeSize = len(code),
			pCode    = (^u32)(raw_data(code)),
		}
		if vk.CreateShaderModule(device, &createInfo, nil, &shaderModule) != .SUCCESS {
			panic("Failed to create shader module")
		}
		return
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
		panic("Failed to create descriptor pool!")
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
		panic("Failed to create compute descriptor set layout!")
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
		panic("Failed to allocate compute descriptor sets!")
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
		panic("Failed to create postprocess pipeline layout!")
	}

	shaderStageInfo: vk.PipelineShaderStageCreateInfo = {
		sType               = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		pNext               = nil,
		flags               = {},
		stage               = {.COMPUTE},
		module              = createShaderModule(fmt.aprintf("./{}", shaderPath)),
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
		panic("Failed to create postprocess pipeline!")
	}

	vk.DestroyShaderModule(device, shaderStageInfo.module, nil)
}

cleanupPipeline :: proc() {
	vk.DestroyDescriptorPool(device, descriptorPool, nil)
	vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)
	vk.DestroyPipeline(device, pipeline, nil)
	vk.DestroyPipelineLayout(device, pipelineLayout, nil)
}

recordComputeBuffer :: proc(commandBuffer: vk.CommandBuffer, imageIndex: u32) {
	beginInfo: vk.CommandBufferBeginInfo = {
		sType            = .COMMAND_BUFFER_BEGIN_INFO,
		pNext            = nil,
		flags            = {},
		pInheritanceInfo = nil,
	}

	if vk.BeginCommandBuffer(commandBuffer, &beginInfo) != .SUCCESS {
		panic("Failed to start recording compute commands!")
	}

	transitionImageLayout(
		commandBuffer,
		inImage.image,
		inImage.format,
		.UNDEFINED,
		.TRANSFER_DST_OPTIMAL,
		1,
	)

	transitionImageLayout(
		commandBuffer,
		inImage.image,
		inImage.format,
		.TRANSFER_DST_OPTIMAL,
		.GENERAL,
		1,
	)

	transitionImageLayout(
		commandBuffer,
		outImage.image,
		outImage.format,
		.TRANSFER_SRC_OPTIMAL,
		.GENERAL,
		1,
	)

	vk.CmdBindDescriptorSets(commandBuffer, .COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nil)

	vk.CmdBindPipeline(commandBuffer, .COMPUTE, pipeline)

	vk.CmdDispatch(commandBuffer, 100 / 32 + 1, 100 / 32 + 1, 1)

	transitionImageLayout(
		commandBuffer,
		outImage.image,
		outImage.format,
		.UNDEFINED,
		.TRANSFER_SRC_OPTIMAL,
		1,
	)

	if vk.EndCommandBuffer(commandBuffer) != .SUCCESS {
		panic("Failed to record compute command buffer!")
	}
}

cleanupVkGraphics :: proc() {
	vk.DeviceWaitIdle(device)
	vk.DestroyCommandPool(device, commandPool, nil)
	vk.DestroyDevice(device, nil)
	vk.DestroyInstance(instance, nil)
}
