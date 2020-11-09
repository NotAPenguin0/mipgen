#include <mipgen/mipgen.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>

#if MIPGEN_DEBUG
#include <iostream>
#endif

#if MIPGEN_COMPUTE
#define MIPGEN_COMPUTE_WORKGROUP_SIZE 16
#endif

namespace mipgen {

#if MIPGEN_DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
	VkDebugUtilsMessengerCallbackDataEXT const* callback_data, void*) {

	// Only log messages with severity 'warning' or above
	if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		std::cout << callback_data->pMessage << "\n";
	}

	return VK_FALSE;
}
#endif

static std::vector<uint32_t> load_shader_code(std::string_view file_path) {
	std::ifstream file(file_path.data(), std::ios::binary);
	std::string code = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

	std::vector<uint32_t> spv(code.size() / 4);
	std::memcpy(spv.data(), &code[0], code.size());
	return spv;
}

struct Context {
	VkInstance instance = nullptr;
	VkDevice device = nullptr;
	VkPhysicalDevice phys_device = nullptr;
#if MIPGEN_DEBUG
	VkDebugUtilsMessengerEXT debug_messenger = nullptr;
#endif
	VkPhysicalDeviceProperties properties{};
	uint32_t queue_family_index = 0;
	VkQueue queue = nullptr;
	VkCommandPool cmd_pool = nullptr;

	// Pipeline data for our compute shader
#if MIPGEN_COMPUTE
	VkPipeline pipeline = nullptr;
	VkPipelineLayout pipeline_layout = nullptr;
	VkDescriptorSetLayout descr_set_layout = nullptr;
	VkDescriptorPool descr_pool = nullptr;
	VkSampler sampler = nullptr;
#endif

	// Indicats whether this context refers to an existing vulkan context, or is connected to an external context.
	bool owner = false;

	Context() {
		// We're creating our own context, so set this value to true so we deallocate on free
		owner = true;

		VkApplicationInfo app_info{};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pApplicationName = "Mipmap generator";
		app_info.pEngineName = "Andromeda Engine";
		app_info.apiVersion = VK_API_VERSION_1_2;

		VkInstanceCreateInfo instance_info{};
		instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		const char* layer = "VK_LAYER_KHRONOS_validation";
		instance_info.ppEnabledLayerNames = &layer;
		instance_info.enabledLayerCount = 1;
		const char* debug_utils_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
		instance_info.ppEnabledExtensionNames = &debug_utils_extension;
		instance_info.enabledExtensionCount = 1;
		instance_info.pApplicationInfo = &app_info;

		VkResult result = vkCreateInstance(&instance_info, nullptr, &instance);
		assert(result == VK_SUCCESS && "Failed to create vulkan instance.");
		
#if MIPGEN_DEBUG
		{
			// Load function pointer
			auto create_func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			assert(create_func && "VK_EXT_debug_utils not present");

			// Create debug messenger
			VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
			messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
			messenger_info.pfnUserCallback = vk_debug_callback;
			result = create_func(instance, &messenger_info, nullptr, &debug_messenger);
			assert(result == VK_SUCCESS && "Failed to create debug messenger");
		}
#endif
		{
			// Get physical device. For simplicity, we will always use the first dedicated one we can find
			uint32_t device_count;
			result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
			assert(result == VK_SUCCESS && device_count > 0 && "Could not find any vulkan-capable devices");
			std::vector<VkPhysicalDevice> phys_devices(device_count);
			result = vkEnumeratePhysicalDevices(instance, &device_count, phys_devices.data());
			for (VkPhysicalDevice dev : phys_devices) {
				VkPhysicalDeviceProperties props;
				vkGetPhysicalDeviceProperties(dev, &props);
				if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
					phys_device = dev;
					break;
				}
			}
			// Grab device properties
			vkGetPhysicalDeviceProperties(phys_device, &properties);
		}

		// We need a single graphics queue for mipmap generation.
		{
			const float priority = 1.0f;
			VkDeviceQueueCreateInfo queue_info{};
			queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_info.pQueuePriorities = &priority;
			queue_info.queueCount = 1;
			// Get queue family properties
			uint32_t family_count;
			vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &family_count, nullptr);
			std::vector<VkQueueFamilyProperties> families(family_count);
			vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &family_count, families.data());
#if MIPGEN_COMPUTE
			VkQueueFlagBits queue_flags = VK_QUEUE_COMPUTE_BIT;
#else
			VkQueueFlagBits queue_flags = VK_QUEUE_GRAPHICS_BIT;
#endif
			for (int i = 0; i < family_count; ++i) {
				if (families[i].queueFlags & queue_flags) {
					// We found a valid queue
					queue_info.queueFamilyIndex = i;
					queue_family_index = i;
					break;
				}
			}

			// Now that we have the queue create info, we can create the device
			VkDeviceCreateInfo device_info{};
			device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			device_info.queueCreateInfoCount = 1;
			device_info.pQueueCreateInfos = &queue_info;
			result = vkCreateDevice(phys_device, &device_info, nullptr, &device);
			assert(result == VK_SUCCESS && "Failed to create device");

			// Finally, grab the queue we requested
			vkGetDeviceQueue(device, queue_family_index, 0, &queue);
		}

		create_command_pool();
#if MIPGEN_COMPUTE
		create_compute_pipeline();
#endif
	}

	Context(VkInstance inst, VkDevice dev, VkPhysicalDevice phys_dev, VkQueue q, uint32_t family) 
		: instance(inst), device(dev), phys_device(phys_dev), queue(q), queue_family_index(family) {
		create_command_pool();
#if MIPGEN_COMPUTE
		create_compute_pipeline();
#endif
	}

	~Context() {
		vkDestroyCommandPool(device, cmd_pool, nullptr);
		vkDestroyDescriptorPool(device, descr_pool, nullptr);
		vkDestroyDescriptorSetLayout(device, descr_set_layout, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroySampler(device, sampler, nullptr);
		if (owner) {
#if MIPGEN_DEBUG
			auto destroy_func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			destroy_func(instance, debug_messenger, nullptr);
#endif
			vkDestroyDevice(device, nullptr);
			vkDestroyInstance(instance, nullptr);
		}
	}
private:
	void create_command_pool() {
		VkCommandPoolCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		info.queueFamilyIndex = queue_family_index;
		info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		VkResult result = vkCreateCommandPool(device, &info, nullptr, &cmd_pool);
		assert(result == VK_SUCCESS && "Failed to create command pool");
	}
#if MIPGEN_COMPUTE
	void create_compute_pipeline() {
		// Create descriptor pool
		{
			VkDescriptorPoolCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
			VkDescriptorPoolSize pool_sizes[]{
				{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 32},
				{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32}
			};
			info.pPoolSizes = pool_sizes;
			info.poolSizeCount = sizeof(pool_sizes) / sizeof(VkDescriptorPoolSize);
			info.maxSets = 32; // We'll need one set per miplevel eventually
			VkResult result = vkCreateDescriptorPool(device, &info, nullptr, &descr_pool);
			assert(result == VK_SUCCESS && "Failed to create descriptor pool");
		}
		// Descriptor set layout
		{
			VkDescriptorSetLayoutCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			VkDescriptorSetLayoutBinding bindings[]{
				{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
				{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
			};
			info.pBindings = bindings;
			info.bindingCount = sizeof(bindings) / sizeof(VkDescriptorSetLayoutBinding);
			VkResult result = vkCreateDescriptorSetLayout(device, &info, nullptr, &descr_set_layout);
			assert(result == VK_SUCCESS && "Failed to create descriptor set layout");
		}
		// Pipeline layout
		{
			VkPipelineLayoutCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			info.pushConstantRangeCount = 0;
			info.pPushConstantRanges = nullptr;
			info.setLayoutCount = 1;
			info.pSetLayouts = &descr_set_layout;
			VkResult result = vkCreatePipelineLayout(device, &info, nullptr, &pipeline_layout);
		}
		// Pipeline
		{
			// 1. Shader module
			auto code = load_shader_code("mipgen.spv");
			VkShaderModuleCreateInfo module_info{};
			module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			module_info.codeSize = code.size() * 4;
			module_info.pCode = code.data();
			VkShaderModule sh_module = nullptr;
			VkResult result = vkCreateShaderModule(device, &module_info, nullptr, &sh_module);
			assert(result == VK_SUCCESS && "Failed to create shader module");

			// 2. Pipeline
			VkComputePipelineCreateInfo pci{};
			pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pci.basePipelineHandle = nullptr;
			pci.basePipelineIndex = 0;
			pci.layout = pipeline_layout;
			pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			pci.stage.module = sh_module;
			pci.stage.pName = "main";
			pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			
			result = vkCreateComputePipelines(device, nullptr, 1, &pci, nullptr, &pipeline);
			assert(result == VK_SUCCESS && "Could not create compute pipeline");

			vkDestroyShaderModule(device, sh_module, nullptr);
		}
		// Image sampler 
		{
			VkSamplerCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			info.anisotropyEnable = false;
			info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			info.compareEnable = false;
			info.magFilter = VK_FILTER_NEAREST;
			info.maxLod = 0.0f;
			info.minFilter = VK_FILTER_NEAREST;
			info.mipLodBias = 0.0f;
			info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			info.unnormalizedCoordinates = false;
			VkResult result = vkCreateSampler(device, &info, nullptr, &sampler);
			assert(result == VK_SUCCESS && "Could not create sampler");
		}
	}
#endif
};

Context* create_context() {
	return new Context;
}

Context* create_context(uint64_t instance, uint64_t device, uint64_t phys_device, uint64_t queue, uint32_t family_index) {
	return new Context(reinterpret_cast<VkInstance>(instance), 
		reinterpret_cast<VkDevice>(device), 
		reinterpret_cast<VkPhysicalDevice>(phys_device),
		reinterpret_cast<VkQueue>(queue), family_index);
}

void free_context(Context* ctx) {
	delete ctx;
}

uint32_t get_mip_count(ImageInfo const& info) {
	// We won't count 3D for miplevel generation (why would you even).
	return std::log2(std::max(info.extents[0], info.extents[1])) + 1;
}

static uint32_t mip_size_x(ImageInfo const& info, uint32_t mip) {
	return (info.extents[0] >> mip);
}

static uint32_t mip_size_y(ImageInfo const& info, uint32_t mip) {
	return (info.extents[1] >> mip);
}

static uint32_t format_byte_size(ImageFormat format) {
	switch (format) {
	case ImageFormat::RGBA8:
	case ImageFormat::sRGBA8:
		return 4;
	default:
		return 0;
	}
}

// Returns VkFormat that fits the original format, but does not respect color space
static VkFormat get_vk_format(ImageFormat format) {
	switch (format) {
	case ImageFormat::RGBA8:
	case ImageFormat::sRGBA8:
		return VK_FORMAT_R8G8B8A8_UNORM;
	}
}

// Returns matching VkFormat that does respect color space.
static VkFormat get_vk_format_srgb(ImageFormat format) {
	switch (format) {
	case ImageFormat::sRGBA8:
		return VK_FORMAT_R8G8B8A8_SRGB;
	case ImageFormat::RGBA8:
		return VK_FORMAT_R8G8B8A8_UNORM;
	}
}

uint32_t output_buffer_size(ImageInfo const& info) {
	// Total amount of pixels
	uint32_t pixels = 0;
	for (uint32_t level = 0; level < get_mip_count(info); ++level) {
		pixels += (info.extents[0] / pow(2, level)) * (info.extents[1] / pow(2, level));
	}
	return pixels * format_byte_size(info.format);
}

uint32_t find_memory_type(Context* ctx, uint32_t type_filter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(ctx->phys_device, &mem_props);
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties)) {
			return i;
		}
	}
	assert(false && "Failed to find suitable memory type");
	return -1;
}

static VkCommandBuffer create_command_buffer(Context* ctx) {
	VkCommandBuffer cmd_buf = nullptr;
	VkCommandBufferAllocateInfo cmdbuf_info{};
	cmdbuf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdbuf_info.commandBufferCount = 1;
	cmdbuf_info.commandPool = ctx->cmd_pool;
	cmdbuf_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	VkResult result = vkAllocateCommandBuffers(ctx->device, &cmdbuf_info, &cmd_buf);
	assert(result == VK_SUCCESS && "Failed to allocate command buffer");

	VkCommandBufferBeginInfo begin_info{};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd_buf, &begin_info);
	return cmd_buf;
}

// Submits the command buffer to our queue and awaits its completion using a fence. Then it frees the command buffer.
static void submit_and_wait(Context* ctx, VkCommandBuffer cmd_buf) {
	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd_buf;
	submit.waitSemaphoreCount = 0;
	submit.signalSemaphoreCount = 0;

	VkFenceCreateInfo fence_info{};
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkFence fence = nullptr;
	VkResult result = vkCreateFence(ctx->device, &fence_info, nullptr, &fence);
	assert(result == VK_SUCCESS && "Could not create fence");

	result = vkQueueSubmit(ctx->queue, 1, &submit, fence);
	assert(result == VK_SUCCESS && "Could not submit command buffer");
	vkWaitForFences(ctx->device, 1, &fence, true, std::numeric_limits<uint64_t>::max());
	vkDestroyFence(ctx->device, fence, nullptr);
	vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
}

namespace {
struct RawImage {
	VkImage image = nullptr;
	VkDeviceMemory memory = nullptr;
};

struct RawBuffer {
	VkBuffer buffer = nullptr;
	void* pointer = nullptr;
	VkDeviceMemory memory = nullptr;
	VkDeviceSize size = 0;
};
}

// Creates a VkImage with mip_count miplevels, and uploads image_info.pixels to the first miplevel.
// This function also creates a staging buffer with persistently mapped memory the size of the full image. 
// This buffer can be used to read back data from the image later.
// The function inserts a barrier that properly synchronizes with the upcoming compute/transfer calls and transitions the image to VK_IMAGE_LAYOUT_GENERAL
static VkResult create_and_upload_image(Context* ctx, VkCommandBuffer cmd_buf, RawImage& image, RawBuffer& buffer, ImageInfo const& image_info) {
	uint32_t mip_count = get_mip_count(image_info);
	// Create the image. Fairly standard stuff
	VkImageCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType = VK_IMAGE_TYPE_2D;
	info.extent.width = image_info.extents[0];
	info.extent.height = image_info.extents[1];
	info.extent.depth = 1;
	info.arrayLayers = 1;
	info.format = get_vk_format(image_info.format);
	info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	info.mipLevels = mip_count;
	info.tiling = VK_IMAGE_TILING_OPTIMAL;
	info.samples = VK_SAMPLE_COUNT_1_BIT;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
#if MIPGEN_COMPUTE
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
#else
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
#endif
	VkResult result = vkCreateImage(ctx->device, &info, nullptr, &image.image);
	if (result != VK_SUCCESS) return result;

	// Allocate memory for the image. Also fairly standard
	VkMemoryRequirements mem_reqs{};
	vkGetImageMemoryRequirements(ctx->device, image.image, &mem_reqs);
	VkMemoryAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_reqs.size;
	alloc_info.memoryTypeIndex = find_memory_type(ctx, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	result = vkAllocateMemory(ctx->device, &alloc_info, nullptr, &image.memory);
	if (result != VK_SUCCESS) return result;
	result = vkBindImageMemory(ctx->device, image.image, image.memory, 0);
	if (result != VK_SUCCESS) return result;

	// Now we create a staging buffer so we can copy our data to the image
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	buffer_info.size = output_buffer_size(image_info);
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	buffer.size = buffer_info.size;
	result = vkCreateBuffer(ctx->device, &buffer_info, nullptr, &buffer.buffer);
	if (result != VK_SUCCESS) return result;
	vkGetBufferMemoryRequirements(ctx->device, buffer.buffer, &mem_reqs);
	alloc_info.memoryTypeIndex = find_memory_type(ctx, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	result = vkAllocateMemory(ctx->device, &alloc_info, nullptr, &buffer.memory);
	if (result != VK_SUCCESS) return result;
	result = vkBindBufferMemory(ctx->device, buffer.buffer, buffer.memory, 0);
	if (result != VK_SUCCESS) return result;
	result = vkMapMemory(ctx->device, buffer.memory, 0, buffer.size, 0, &buffer.pointer);
	if (result != VK_SUCCESS) return result;
	// Copy over pixel data
	uint32_t base_layer_bytes = image_info.extents[0] * image_info.extents[1] * format_byte_size(image_info.format);
	memcpy(buffer.pointer, image_info.pixels, base_layer_bytes);

	// Transition image to GENERAL (easier to work with)
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image.image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = mip_count;
	barrier.srcAccessMask = {};
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	vkCmdPipelineBarrier(cmd_buf,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &barrier);

	// Copy staging buffer data (only base layer) to image
	VkBufferImageCopy copy{};
	copy.bufferRowLength = image_info.extents[0];
	copy.bufferImageHeight = image_info.extents[1];
	copy.bufferOffset = 0;
	copy.imageExtent.width = image_info.extents[0];
	copy.imageExtent.height = image_info.extents[1];
	copy.imageExtent.depth = 1;
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.baseArrayLayer = 0;
	copy.imageSubresource.layerCount = 1;
	copy.imageSubresource.mipLevel = 0;
	copy.imageOffset.x = copy.imageOffset.y = copy.imageOffset.z = 0;
	vkCmdCopyBufferToImage(cmd_buf, buffer.buffer, image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
	// Make sure write is complete before starting read (or in compute case, compute shader read)
	barrier.subresourceRange.levelCount = 1;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
#if MIPGEN_COMPUTE
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
#else
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
#endif
	barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
#if MIPGEN_COMPUTE
	vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &barrier);
#else
	vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &barrier);
#endif
	return VK_SUCCESS;
}

static void destroy_image(Context* ctx, RawImage& image) {
	vkFreeMemory(ctx->device, image.memory, nullptr);
	vkDestroyImage(ctx->device, image.image, nullptr);

	image.memory = nullptr;
	image.image = nullptr;
}

static void destroy_buffer(Context* ctx, RawBuffer& buffer) {
	vkUnmapMemory(ctx->device, buffer.memory);
	vkFreeMemory(ctx->device, buffer.memory, nullptr);
	vkDestroyBuffer(ctx->device, buffer.buffer, nullptr);

	buffer.memory = nullptr;
	buffer.buffer = nullptr;
	buffer.size = 0;
	buffer.pointer = nullptr;
}

#if MIPGEN_COMPUTE

uint32_t generate_mipmap_compute(Context* ctx, ImageInfo const& image_info, void* out_buffer) {
	uint32_t const mip_count = get_mip_count(image_info);
	uint32_t const output_size = output_buffer_size(image_info);

	VkCommandBuffer cmd_buf = create_command_buffer(ctx);

	RawImage image{};
	RawBuffer staging{};
	{
		VkResult result = create_and_upload_image(ctx, cmd_buf, image, staging, image_info);
		assert(result == VK_SUCCESS && "Failed to create/upload image");
	}

	// Next we're going to create an image view for every mip level, both with format sRGB and UNORM.
	// The reason for this is that we can't store to sRGB images, but we have to sample in the correct color space.
	std::vector<VkImageView> views;
	std::vector<VkImageView> srgb_views;
	for (uint32_t mip = 0; mip < mip_count; ++mip) {
		VkImageViewCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		info.components.r = VK_COMPONENT_SWIZZLE_R;
		info.components.g = VK_COMPONENT_SWIZZLE_G;
		info.components.b = VK_COMPONENT_SWIZZLE_B;
		info.components.a = VK_COMPONENT_SWIZZLE_A;
		info.format = get_vk_format(image_info.format);
		info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		info.subresourceRange.baseArrayLayer = 0;
		info.subresourceRange.layerCount = 1;
		info.subresourceRange.baseMipLevel = mip;
		info.subresourceRange.levelCount = 1;
		info.image = image.image;
		
		VkImageView unorm_view = nullptr;
		VkResult result = vkCreateImageView(ctx->device, &info, nullptr, &unorm_view);
		assert(result == VK_SUCCESS && "Could not create image view");
		views.push_back(unorm_view);

		VkImageView srgb_view = nullptr;
		// Since this is going to be an sRGB format, we need to further retrict the VkImageUsageFlags of this specific VkImageView.
		// In Vulkan 1.1 or up, we can do this by adding VkImageViewUsageCreateInfo to the pNext chain of VkImageViewCreateInfo
		VkImageViewUsageCreateInfo usage{};
		usage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
		usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT; // We only sample from this image view
		usage.pNext = nullptr;
		info.pNext = &usage;
		info.format = get_vk_format_srgb(image_info.format);
		result = vkCreateImageView(ctx->device, &info, nullptr, &srgb_view);
		assert(result == VK_SUCCESS && "Could not create image view");
		srgb_views.push_back(srgb_view);
	}

	vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);

	std::vector<VkDescriptorSet> descr_sets; // We store all descriptor sets so we can free them when we are done
	for (uint32_t mip = 1; mip < mip_count; ++mip) {
		// Create descriptor sets pointing at the correct image views
		VkDescriptorSetAllocateInfo set_info{};
		set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		set_info.descriptorPool = ctx->descr_pool;
		set_info.descriptorSetCount = 1;
		set_info.pSetLayouts = &ctx->descr_set_layout;
		VkDescriptorSet descr_set = nullptr;
		vkAllocateDescriptorSets(ctx->device, &set_info, &descr_set);
		descr_sets.push_back(descr_set);

		// Update descriptor set accordingly
		VkWriteDescriptorSet writes[2];
		// input image
		writes[0].pNext = nullptr;
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[0].dstArrayElement = 0;
		writes[0].dstBinding = 0;
		writes[0].dstSet = descr_set;
		VkDescriptorImageInfo in_image_info{};
		in_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		in_image_info.imageView = srgb_views[mip - 1]; // This one is sampled so it needs to be in the correct colorspace.
		in_image_info.sampler = ctx->sampler;
		writes[0].pImageInfo = &in_image_info;
		writes[0].pTexelBufferView = nullptr;
		writes[0].pBufferInfo = nullptr;
		// output image
		writes[1].pNext = nullptr;
		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[1].dstArrayElement = 0;
		writes[1].dstBinding = 1;
		writes[1].dstSet = descr_set;
		VkDescriptorImageInfo out_image_info{};
		out_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		out_image_info.imageView = views[mip]; // Not sampled, only written to. Use raw format without colorspace information.
		out_image_info.sampler = ctx->sampler;
		writes[1].pImageInfo = &out_image_info;
		writes[1].pTexelBufferView = nullptr;
		writes[1].pBufferInfo = nullptr;

		vkUpdateDescriptorSets(ctx->device, 2, writes, 0, nullptr);
		vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &descr_set, 0, nullptr);

		// Dispatch this miplevel
		uint32_t dispatches_x = std::ceil(mip_size_x(image_info, mip) / (float)MIPGEN_COMPUTE_WORKGROUP_SIZE);
		uint32_t dispatches_y = std::ceil(mip_size_y(image_info, mip) / (float)MIPGEN_COMPUTE_WORKGROUP_SIZE);
		vkCmdDispatch(cmd_buf, dispatches_x, dispatches_y, 1);

		// Add pipeline barrier to protect memory from current miplevel being read by the next miplevel before next writing is done
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.image = image.image;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.baseMipLevel = mip;
		barrier.subresourceRange.levelCount = 1;
		vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
			VK_DEPENDENCY_BY_REGION_BIT,
			0, nullptr, 0, nullptr, 1, &barrier);
	}

	// Add one more barrier protecting the whole image before we do our readback to the staging buffer
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.image = image.image;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = mip_count;
	vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_DEPENDENCY_BY_REGION_BIT,
		0, nullptr, 0, nullptr, 1, &barrier);

	// Copy image data to our staging buffer
	uint32_t buffer_offset = 0;
	for (uint32_t mip = 0; mip < mip_count; ++mip) {
		VkBufferImageCopy copy{};
		copy.bufferRowLength = mip_size_x(image_info, mip);
		copy.bufferImageHeight = mip_size_y(image_info, mip);
		copy.bufferOffset = buffer_offset;
		copy.imageExtent.width = mip_size_x(image_info, mip);
		copy.imageExtent.height = mip_size_y(image_info, mip);
		copy.imageExtent.depth = 1;
		copy.imageOffset.x = copy.imageOffset.y = copy.imageOffset.z = 0;
		copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copy.imageSubresource.baseArrayLayer = 0;
		copy.imageSubresource.layerCount = 1;
		copy.imageSubresource.mipLevel = mip;
		vkCmdCopyImageToBuffer(cmd_buf, image.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &copy);
		int32_t miplevel_byte_size = (mip_size_x(image_info, mip)) * (mip_size_y(image_info, mip)) * format_byte_size(image_info.format);
		buffer_offset += miplevel_byte_size;
	}

	// Submit command buffer
	vkEndCommandBuffer(cmd_buf);
	submit_and_wait(ctx, cmd_buf);

	// Read back pixels
	memcpy(out_buffer, staging.pointer, output_size);

	for (VkImageView view : views) {
		vkDestroyImageView(ctx->device, view, nullptr);
	}
	for (VkImageView view : srgb_views) {
		vkDestroyImageView(ctx->device, view, nullptr);
	}
	vkFreeDescriptorSets(ctx->device, ctx->descr_pool, descr_sets.size(), descr_sets.data());
	destroy_image(ctx, image);
	destroy_buffer(ctx, staging);

	return output_size;
}

#else

static uint32_t generate_mipmap_graphics(Context* ctx, ImageInfo const& image_info, void* out_buffer) {
	// We need to do a few things:
	// 1. Create the command buffer
	// 2. Upload the image pixels to a GPU image
	// 3. Record commands for mipmap generation
	// 4. Submit command buffer
	// 5. Read back pixels from image.
	// 6. Cleanup

	uint32_t const mip_count = get_mip_count(image_info);
	uint32_t const output_size = output_buffer_size(image_info);

	// 1. Create command buffer 
	VkCommandBuffer cmd_buf = create_command_buffer(ctx);

	// 2. Create image, upload pixel data
	RawImage image{};
	RawBuffer staging{};
	{
		VkResult result = create_and_upload_image(ctx, cmd_buf, image, staging, image_info);
		assert(result == VK_SUCCESS && "Failed to create/upload image");
	}

	// 3. Record commands for mipmap generation
	{
		// See also https://github.com/SaschaWillems/Vulkan/blob/e370e6d169204bc3deaef637189336972414ffa5/examples/texturemipmapgen/texturemipmapgen.cpp#L264
		// for this algorithm
		for (uint32_t mip = 1; mip < mip_count; ++mip) {
			VkImageBlit blit{};
			// Source info
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.layerCount = 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.mipLevel = mip - 1;
			blit.srcOffsets[1].x = (int32_t)(image_info.extents[0] >> (mip - 1));
			blit.srcOffsets[1].y = (int32_t)(image_info.extents[1] >> (mip - 1));
			blit.srcOffsets[1].z = 1;
			// Destination info
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.layerCount = 1;
			blit.dstSubresource.mipLevel = mip;
			blit.dstOffsets[1].x = (int32_t)(image_info.extents[0] >> mip);
			blit.dstOffsets[1].y = (int32_t)(image_info.extents[0] >> mip);
			blit.dstOffsets[1].z = 1;

			// Do the blit
			vkCmdBlitImage(cmd_buf, image.image, VK_IMAGE_LAYOUT_GENERAL, image.image, VK_IMAGE_LAYOUT_GENERAL,
				1, &blit, VK_FILTER_LINEAR);

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image.image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.baseMipLevel = mip;
			barrier.subresourceRange.levelCount = 1;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_BY_REGION_BIT,
				0, nullptr, 0, nullptr, 1, &barrier);
		}
		vkEndCommandBuffer(cmd_buf);
		// TODO: Condense into 1 command buffer and add barrier
	}

	// 4. Submit command buffer
	submit_and_wait(ctx, cmd_buf);

	// 5. Write pixels back into output buffer
	{
		// 5a. Create new command buffer
		VkCommandBufferAllocateInfo cmdbuf_info{};
		cmdbuf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdbuf_info.commandBufferCount = 1;
		cmdbuf_info.commandPool = ctx->cmd_pool;
		cmdbuf_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		VkResult result = vkAllocateCommandBuffers(ctx->device, &cmdbuf_info, &cmd_buf);
		assert(result == VK_SUCCESS && "Failed to allocate command buffer");

		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd_buf, &begin_info);

		// 5b. Read back pixels, miplevel by miplevel
		uint32_t buffer_offset = 0;
		for (uint32_t mip = 0; mip < mip_count; ++mip) {
			VkBufferImageCopy copy{};
			copy.bufferRowLength = (int32_t)(image_info.extents[0] >> mip);
			copy.bufferImageHeight = (int32_t)(image_info.extents[1] >> mip);
			copy.bufferOffset = buffer_offset;
			copy.imageExtent.width = (int32_t)(image_info.extents[0] >> mip);
			copy.imageExtent.height = (int32_t)(image_info.extents[1] >> mip);
			copy.imageExtent.depth = 1;
			copy.imageOffset.x = copy.imageOffset.y = copy.imageOffset.z = 0;
			copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copy.imageSubresource.baseArrayLayer = 0;
			copy.imageSubresource.layerCount = 1;
			copy.imageSubresource.mipLevel = mip;
			vkCmdCopyImageToBuffer(cmd_buf, image.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &copy);
			int32_t miplevel_byte_size = (image_info.extents[0] / pow(2, mip)) * (image_info.extents[1] / pow(2, mip)) * format_byte_size(image_info.format);
			buffer_offset += miplevel_byte_size;
		}
		vkEndCommandBuffer(cmd_buf);

		// 5c. Submit command buffer
		submit_and_wait(ctx, cmd_buf);

		// 5d. Copy staging buffer memory to output pointer
		memcpy(out_buffer, staging.pointer, output_size);
	}

	// 6. Cleanup
	destroy_image(ctx, image);
	destroy_buffer(ctx, buffer);

	return output_size;
}

#endif

uint32_t generate_mipmap(Context* ctx, ImageInfo const& image_info, void* out_buffer) {
#if MIPGEN_COMPUTE
	return generate_mipmap_compute(ctx, image_info, out_buffer);
#else
	return generate_mipmap_graphics(ctx, image_info, out_buffer);
#endif
}

}