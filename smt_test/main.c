#include <stdint.h>

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define ABS(X) ((X) > 0 ? (X) : (-(X)))

// Riscv-Doom Like VGA
#define WIDTH 640
#define HEIGHT 480
#define FRAMESIZE (WIDTH*HEIGHT)
static volatile unsigned int*  VGA_CONTROL_BASE =  (unsigned int* )0x81000000u;
static volatile unsigned int*  VGA_PALLETE_BASE =  (unsigned int* )0x81010000u;
static volatile unsigned char* VGA_SCREEN_0_BASE = (unsigned char*)0x81020000u;
static volatile unsigned char* VGA_SCREEN_1_BASE = (unsigned int*)(0x81020000u + FRAMESIZE);

// "FAKE" DMA Controller Interface
static volatile unsigned int* dmaDstAddr =    (unsigned int*)0x65000000u;
static volatile unsigned int* dmaSrcAddr =    (unsigned int*)0x65000004u;
static volatile unsigned int* dmaTransferSz = (unsigned int*)0x65000008u;
static volatile unsigned int* dmaBusy =       (unsigned int*)0x6500000cu;
static volatile unsigned int* dmaCommit =     (unsigned int*)0x65000010u;

// "FAKE" GPU Interface
static volatile uint32_t* const REG_RB_BASE  = (volatile uint32_t*)(0x64000000u);
static volatile uint32_t* const REG_RB_SIZE  = (volatile uint32_t*)(0x64000004u);
static volatile uint32_t* const REG_RB_RPTR  = (volatile uint32_t*)(0x64000008u);
static volatile uint32_t* const REG_RB_WPTR  = (volatile uint32_t*)(0x6400000Cu);
static volatile uint32_t* const REG_STATUS   = (volatile uint32_t*)(0x64000010u);

// Command opcodes (32-bit headers)
// Format: [8-bit Opcode] [24-bit Count of following words]
#define CMD_NOP             0x00  // Body: none
#define CMD_SET_VBO         0x01  // Body: [Addr] [Stride] [Format] (16.16 fixed-point 3D)
#define CMD_DRAW            0x02  // Body: [VertexCount]
#define CMD_FENCE           0x03  // Body: [WriteAddress] [WriteValue]
#define CMD_SET_VERTEX_SHADER    0x04  // Body: [ShaderAddr] [ShaderSize]
#define CMD_SET_FRAGMENT_SHADER  0x05  // Body: [ShaderAddr] [ShaderSize]
#define CMD_SET_UNIFORM          0x06  // Body: [UniformID] [Value0] [Value1] [Value2] [Value3]
#define CMD_SET_UNIFORM_MAT4     0x07  // Body: [UniformID] [16 values]
#define CMD_SET_TEXTURE          0x08  // Body: [TextureUnit] [TextureAddr] [Width] [Height] [Format]
#define CMD_CLEAR                0x09  // Body: [ClearFlags] [Color_RGBA]

// Helper to build packet header
#define PKT_HEADER(op, count) (((op) << 24) | ((count) & 0xFFFFFF))

// Vertex buffer format
#define VBO_FORMAT_FIXED16_16_3D 0x0

// Uniform IDs
#define UNIFORM_MODEL_MATRIX      0
#define UNIFORM_VIEW_MATRIX       1
#define UNIFORM_PROJ_MATRIX       2
#define UNIFORM_TIME              3
#define UNIFORM_LIGHT_POS         4
#define UNIFORM_CAMERA_POS        5

// Clear flags
#define CLEAR_COLOR_BUFFER  (1 << 0)
#define CLEAR_DEPTH_BUFFER  (1 << 1)
#define CLEAR_STENCIL_BUFFER (1 << 2)

// GPU driver context
typedef struct {
    uint32_t* ring_cpu_addr; // CPU-visible ring buffer
    uint32_t  wptr;          // CPU write pointer (in bytes)
    uint32_t  mask;          // (size_bytes - 1) for fast wraparound
} gpu_ctx_t;

// Initialize GPU and ring buffer
static inline void gpu_init(gpu_ctx_t* ctx, volatile uint32_t* ring_mem, uint32_t ring_size_words) {
    ctx->ring_cpu_addr = (uint32_t*)ring_mem;
    ctx->wptr = 0;
    uint32_t size_bytes = ring_size_words * 4;
    ctx->mask = size_bytes - 1;  // Bitmask for fast wraparound

    *REG_RB_BASE = (uint32_t)(ring_mem);
    *REG_RB_SIZE = size_bytes;  // GPU expects bytes
    *REG_RB_WPTR = 0;

    __sync_synchronize();
}

// Write a word into the ring buffer
static inline void emit_word(gpu_ctx_t* ctx, uint32_t word) {
    uint32_t byte_offset = ctx->wptr & ctx->mask;  // Fast wraparound with bitmask
    uint32_t word_index = byte_offset >> 2;        // Divide by 4 (shift right 2)
    ctx->ring_cpu_addr[word_index] = word;
    ctx->wptr = (ctx->wptr + 4) & ctx->mask;       // Increment and wrap
}

// Commit CPU commands to GPU
static inline void gpu_commit(gpu_ctx_t* ctx) {
    __sync_synchronize();
    *REG_RB_WPTR = ctx->wptr;  // Already wrapped in emit_word
}

// Set current vertex buffer
static inline void cmd_set_vbo(gpu_ctx_t* ctx, uint32_t phys_addr, uint32_t stride_bytes, uint32_t format) {
    emit_word(ctx, PKT_HEADER(CMD_SET_VBO, 3));
    emit_word(ctx, (uint32_t)phys_addr);
    emit_word(ctx, stride_bytes);
    emit_word(ctx, format);
}

// Draw call
static inline void cmd_draw(gpu_ctx_t* ctx, uint32_t vertex_count) {
    emit_word(ctx, PKT_HEADER(CMD_DRAW, 1));
    emit_word(ctx, vertex_count);
}

// Fence command
static inline void cmd_fence(gpu_ctx_t* ctx, uint32_t fence_addr_phys, uint32_t fence_val) {
    emit_word(ctx, PKT_HEADER(CMD_FENCE, 2));
    emit_word(ctx, (uint32_t)fence_addr_phys);
    emit_word(ctx, fence_val);
}

// Wait for fence
static inline void gpu_wait_fence(volatile uint32_t* fence_addr_virt, uint32_t target_val) {
    while (*fence_addr_virt < target_val) { }
}

// Set vertex shader program
static inline void cmd_set_vertex_shader(gpu_ctx_t* ctx, uint32_t shader_addr, uint32_t shader_size) {
    emit_word(ctx, PKT_HEADER(CMD_SET_VERTEX_SHADER, 2));
    emit_word(ctx, shader_addr);
    emit_word(ctx, shader_size);  // Size in bytes
}

// Set fragment shader program
static inline void cmd_set_fragment_shader(gpu_ctx_t* ctx, uint32_t shader_addr, uint32_t shader_size) {
    emit_word(ctx, PKT_HEADER(CMD_SET_FRAGMENT_SHADER, 2));
    emit_word(ctx, shader_addr);
    emit_word(ctx, shader_size);
}

// Set a single vec4 uniform
static inline void cmd_set_uniform_vec4(gpu_ctx_t* ctx, uint32_t uniform_id, 
                                        int32_t x, int32_t y, int32_t z, int32_t w) {
    emit_word(ctx, PKT_HEADER(CMD_SET_UNIFORM, 5));
    emit_word(ctx, uniform_id);
    emit_word(ctx, x);
    emit_word(ctx, y);
    emit_word(ctx, z);
    emit_word(ctx, w);
}

// Set a mat4 uniform (16 values)
static inline void cmd_set_uniform_mat4(gpu_ctx_t* ctx, uint32_t uniform_id, int32_t* matrix) {
    emit_word(ctx, PKT_HEADER(CMD_SET_UNIFORM_MAT4, 17));
    emit_word(ctx, uniform_id);
    for (int i = 0; i < 16; i++) {
        emit_word(ctx, matrix[i]);
    }
}

// Set texture
static inline void cmd_set_texture(gpu_ctx_t* ctx, uint32_t texture_unit, 
                                    uint32_t texture_addr, uint32_t width, uint32_t height) {
    emit_word(ctx, PKT_HEADER(CMD_SET_TEXTURE, 4));
    emit_word(ctx, texture_unit);
    emit_word(ctx, texture_addr);
    emit_word(ctx, width);
    emit_word(ctx, height);
}

// Clear buffers
static inline void cmd_clear(gpu_ctx_t* ctx, uint32_t clear_flags, uint32_t color_rgba) {
    emit_word(ctx, PKT_HEADER(CMD_CLEAR, 2));
    emit_word(ctx, clear_flags);
    emit_word(ctx, color_rgba);
}

// Dual FB setup
static unsigned char* fb0;
static unsigned char* fb1;

// Riscv-Doom Like UART
static volatile char* UART_BASE = 0x82000000u;

#define LOG_RENDER 1

// print string to UART
void _puts(const char* str){
    while(*str){
        *UART_BASE = *str++;
    }
}

void delay(uint32_t n){
    for (uint32_t i = 0; i < n*1024; i++)
    {
        *UART_BASE = 0;
    }
}

// prevent include of stdio.h
#define puts _puts

// Execute a dma transfer from dst to src of size bytes
void dma_transfer(void* dst, void* src, unsigned int size){
    // wait dma busy
    while(*dmaBusy);

    // setup transfer
    *dmaDstAddr = (unsigned int)dst;
    *dmaSrcAddr = (unsigned int)src;
    *dmaTransferSz = size;

    // commit
    *dmaCommit = 1;
}

char* _itoa_10 (int value, char* str){
    char buf[64] = {0};
    char* pBuf = buf;
    int n = 0;
    if(value < 0){
        *(str++) = '-';
    }
    else if(value == 0){
        *str = '0';
        return str;
    }
    while (value != 0)
    {
        *(pBuf++) = '0' + (ABS(value) % 10);
        value /= 10;
        n++;
    }
    n-=1;
    while(n>-1){
        *str++ = buf[n];
        n--;
    }
    *str = '\0';
    return str;
}

char* _itoa_10_pad(int value, char* str, int pad) {
    char buf[64] = {0};
    char* pBuf = buf;
    int n = 0;
    int is_negative = 0;

    if (value < 0) {
        is_negative = 1;
        value = -value;
    } else if (value == 0) {
        // Pad with leading zeros
        for (int i = 0; i < pad; i++)
            *(str++) = '0';
        *str = '\0';
        return str;
    }

    while (value != 0) {
        *(pBuf++) = '0' + (value % 10);
        value /= 10;
        n++;
    }

    int total_digits = (n > pad) ? n : pad;

    if (is_negative)
        *(str++) = '-';

    for (int i = total_digits - n; i > 0; i--)  // leading zero padding
        *(str++) = '0';

    while (--n >= 0)
        *(str++) = buf[n];

    *str = '\0';
    return str;
}

//#undef memset
#undef memcpy

void _memset(void* dst, int val, unsigned int size){
    dma_transfer((void*)dst, (void*)&val, size);
}

void _memcpy(void* dst, void* src, unsigned int size){
    dma_transfer((void*)dst, (void*)src, size);
}

#define memset _memset
#define memcpy _memcpy

typedef struct vec2{
    int32_t e[2];
}vec2;

void pal_write_color(unsigned int color, unsigned char pos){
    VGA_PALLETE_BASE[pos] = color;
}

void wait_vblank(){
    while((*VGA_CONTROL_BASE) >> 16) {};
}

void swap_fb(){
    // copy fb1 to fb0
    wait_vblank();
    memcpy((void*)fb0, (void*)fb1, FRAMESIZE);
    //for (unsigned i = 0; i < FRAMESIZE;){
    //    // Do a dma transfer (line)
    //    dma_transfer((void*)(fb0+i), (void*)(fb1+i), WIDTH);
    //    i += WIDTH;
    //}
}

void clear_fb(unsigned int color){
    // set PALLETE[0] = color
    pal_write_color(color,0);

    // fill FB with 0
    memset(fb1, 0, FRAMESIZE);
}

inline void draw_pixel(int x, int y, unsigned char pal_idx){
    int pos = x + y * (int)WIDTH;
    fb1[pos] = pal_idx;
}

int edge_cross(vec2 a, vec2 b, vec2 p){
    vec2 ab = {{b.e[0] - a.e[0], b.e[1] - a.e[1]}};
    vec2 ap = {{p.e[0] - a.e[0], p.e[1] - a.e[1]}};
    return ab.e[0] * ap.e[1] - ab.e[1] * ap.e[0];
}

void triangle_fill(vec2 v0, vec2 v1, vec2 v2, unsigned int color){
    pal_write_color(color,1);

    // bounding box
    int x_min = MIN(v0.e[0], MIN(v1.e[0], v2.e[0]));
    int y_min = MIN(v0.e[1], MIN(v1.e[1], v2.e[1]));
    int x_max = MAX(v0.e[0], MAX(v1.e[0], v2.e[0]));
    int y_max = MAX(v0.e[1], MAX(v1.e[1], v2.e[1]));

    for (int y = y_min; y < y_max; y++){
        for (int x = x_min; x < x_max; x++){
            vec2 p = {{x,y}};
            int w0 = edge_cross(v1, v2, p);
            int w1 = edge_cross(v2, v0, p);
            int w2 = edge_cross(v0, v1, p);

            if(w0 >= 0 && w1 >= 0 && w2 >= 0){
                draw_pixel(x,y,1);
            }
        }
    }
}

/*
void gpu_barrier(){
    // wait for gpu not busy
    while(*gpuBusy) {};
}

void gpu_commit(){
    *gpuCommit = 1;
}

void gpu_upload_verts(void* vertices_array, unsigned int size){
    // wait for gpu not busy
    gpu_barrier();

    // copy vertices
    memcpy((void*)gpuVertexQueue, vertices_array, size);
    *gpuVertexSz = size / 4;

    gpu_commit();
}

void gpu_upload_vs(void* vs_instructions_array, unsigned int size){
    // wait for gpu not busy
    gpu_barrier();

    // copy vertices
    memcpy((void*)gpuVertexShaderQ, vs_instructions_array, size);
    *gpuVertexShaderSz = size / 4;

    gpu_commit();
}


void gpu_upload_fs(void* fs_instructions_array, unsigned int size){
    // wait for gpu not busy
    gpu_barrier();

    // copy vertices
    memcpy((void*)gpuFragmentShaderQ, fs_instructions_array, size);
    *gpuFragmentShaderSz = size / 4;

    gpu_commit();
}

void gpu_upload_data(void* data_array, unsigned int size){
    // wait for gpu not busy
    gpu_barrier();

    // copy vertices
    memcpy((void*)gpuDataQueue, data_array, size);
    *gpuDataSz = size / 4;

    gpu_commit();
}

*/

#define ONE (1<<16)
#define HALF (1<<15)
#define FIX_MAKE(INT,FRAC) (((int32_t)(INT) << 16) | (FRAC))
static inline int32_t fix16_mul(int32_t x, int32_t y)
{
    int64_t p = (int64_t)x * (int64_t)y;

    // add/sub half for rounding
    if (p >= 0) p += (1LL << 15);
    else        p -= (1LL << 15);

    p >>= 16;

    // saturate to int32_t
    if (p > INT32_MAX) return INT32_MAX;
    if (p < INT32_MIN) return INT32_MIN;

    return (int32_t)p;
}

#define MUL(X,Y) fix16_mul((X),(Y))

static inline int32_t fix16_div(int32_t x, int32_t y)
{
    if (y == 0)
        return (x >= 0) ? INT32_MAX : INT32_MIN;

    int64_t num = (int64_t)x << 16;

    int64_t aby = (y >= 0 ? y : -y);
    int64_t half = aby >> 1;

    // apply rounding based on sign of division result
    if ((num >= 0 && y >= 0) || (num < 0 && y < 0))
        num += half;
    else
        num -= half;

    int64_t r = num / y;

    // saturate
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;

    return (int32_t)r;
}

#define DIV(X,Y) fix16_div((X),(Y))

#define CUBE
#ifdef CUBE
// GPU Shared VBO
static volatile const int32_t g_vertex_buffer_data[] = {
    -ONE,-ONE,-ONE, // triangle 1 : begin
    -ONE,-ONE, ONE,
    -ONE, ONE, ONE, // triangle 1 : end
    ONE, ONE,-ONE, // triangle 2 : begin
    -ONE,-ONE,-ONE,
    -ONE, ONE,-ONE, // triangle 2 : end
    ONE,-ONE, ONE,
    -ONE,-ONE,-ONE,
    ONE,-ONE,-ONE,
    ONE, ONE,-ONE,
    ONE,-ONE,-ONE,
    -ONE,-ONE,-ONE,
    -ONE,-ONE,-ONE,
    -ONE, ONE, ONE,
    -ONE, ONE,-ONE,
    ONE,-ONE, ONE,
    -ONE,-ONE, ONE,
    -ONE,-ONE,-ONE,
    -ONE, ONE, ONE,
    -ONE,-ONE, ONE,
    ONE,-ONE, ONE,
    ONE, ONE, ONE,
    ONE,-ONE,-ONE,
    ONE, ONE,-ONE,
    ONE,-ONE,-ONE,
    ONE, ONE, ONE,
    ONE,-ONE, ONE,
    ONE, ONE, ONE,
    ONE, ONE,-ONE,
    -ONE, ONE,-ONE,
    ONE, ONE, ONE,
    -ONE, ONE,-ONE,
    -ONE, ONE, ONE,
    ONE, ONE, ONE,
    -ONE, ONE, ONE,
    ONE,-ONE, ONE
};
#else
// GPU Shared VBO
static volatile const int32_t g_vertex_buffer_data[] = {
     ONE, -ONE, -ONE,
     ONE,  ONE, -ONE,
    -ONE,  ONE, -ONE,

    -ONE, -ONE, -ONE,
    ONE, -ONE, -ONE,
    -ONE, ONE, -ONE,
};
#endif

#define NVERTS (sizeof(g_vertex_buffer_data) / sizeof(g_vertex_buffer_data[0]))


typedef struct Vec3i
{
    int32_t x, y , z;
}Vec3i;

// GPU Shared VBO
static Vec3i vertices[NVERTS/3] = {0};

typedef int32_t Matrix44i[4][4];

#define IDENTITY { \
    {ONE, 0, 0, 0}, \
    {0, ONE, 0, 0}, \
    {0, 0, ONE, 0}, \
    {0, 0, 0, ONE}  \
}


#include "cordic.h"
#define M_PI PI

void put_digit(int value){
    char* buf[128] = {0};
    _itoa_10(value, buf);
    puts(buf);
}

void put_digit_pad(int value, int pad){
    char* buf[128] = {0};
    _itoa_10_pad(value, buf, pad);
    puts(buf);
}

void put_fixed(int32_t value){
    if(value < 0){
        puts("-");
        value = -value;
    }
    put_digit(value >> 16);
    puts(".");
    put_digit_pad((int)(((uint64_t)(value & 0xffff) * 10000) >> 16),4);
}

// Compute screen coordinates first
void gluPerspective( 
    int32_t angleOfView, 
    int32_t imageAspectRatio, 
    int32_t n, int32_t f, 
    int32_t* b, int32_t* t, int32_t* l, int32_t* r) 
{ 
    int32_t scale = MUL(ONE, n);// tan(angleOfView * 0.5 * M_PI / 180) * n; 
    *r = MUL(imageAspectRatio, scale), *l = -*r; 
    *t = scale, *b = -*t; 
} 
 
// Set the OpenGL perspective projection matrix
void glFrustum( 
    int32_t b, int32_t t, int32_t l, int32_t r, 
    int32_t n, int32_t f, 
    Matrix44i* M) 
{ 
    int32_t twoN = MUL(FIX_MAKE(2,0), n);
    int32_t rml = r-l;
    int32_t rpl = r+l;
    int32_t tmb = t-b;
    int32_t tpb = t+b;
    int32_t fmn = f-n;
    int32_t fpn = f+n;

    // Set OpenGL perspective projection matrix
    (*M)[0][0] = DIV(twoN, rml); 
    (*M)[0][1] = 0; 
    (*M)[0][2] = 0; 
    (*M)[0][3] = 0; 
 
    (*M)[1][0] = 0; 
    (*M)[1][1] = DIV(twoN, tmb); 
    (*M)[1][2] = 0; 
    (*M)[1][3] = 0; 
 
    (*M)[2][0] = DIV(rpl, rml); 
    (*M)[2][1] = DIV(tpb, tmb); 
    (*M)[2][2] = -DIV(fpn, fmn); 
    (*M)[2][3] = -ONE; 
 
    (*M)[3][0] = 0; 
    (*M)[3][1] = 0; 
    (*M)[3][2] = -DIV(MUL(MUL(f,n), FIX_MAKE(2,0)), fmn);
    (*M)[3][3] = 0; 
}
 
void multPointMatrix(Vec3i in, Vec3i* out, Matrix44i M) 
{ 
    //out = in * M;
    out->x   = MUL(in.x, M[0][0]) + MUL(in.y, M[1][0]) + MUL(in.z, M[2][0]) + M[3][0]; 
    out->y   = MUL(in.x, M[0][1]) + MUL(in.y, M[1][1]) + MUL(in.z, M[2][1]) + M[3][1]; 
    out->z   = MUL(in.x, M[0][2]) + MUL(in.y, M[1][2]) + MUL(in.z, M[2][2]) + M[3][2]; 
    int w    = MUL(in.x, M[0][3]) + MUL(in.y, M[1][3]) + MUL(in.z, M[2][3]) + M[3][3]; 

    // normalize if w is different than 1 (convert from homogeneous to Cartesian coordinates)
    if (w != ONE && w != 0) { 
        out->x = DIV(out->x, w); 
        out->y = DIV(out->y, w); 
        out->z = DIV(out->z, w);
    } 
}


static float angle = -M_PI/2.0f;

void testprint(Vec3i vertCamera, Vec3i projectedVert){
    puts("View.z = (");
    put_fixed(vertCamera.z);
    puts(") => (");
    put_fixed(projectedVert.z);
    //puts(") => (");
    //put_fixed(HALF+MUL(HALF,projectedVert.z));
    puts("\n");
}

// Gpu driver
gpu_ctx_t g_gpu_driver;

// GPU Shared Ring buffer in RAM (CPU writes commands here)
// MUST be power of 2!
#define GPU_RING_SIZE 1024
volatile uint32_t g_gpu_ring_ram[GPU_RING_SIZE] __attribute__((aligned(32)));

// GPU Shared Fence variable
volatile uint32_t g_gpu_fence_var = 0;

// Random Shared texture
#define TEX_WIDTH 256
#define TEX_HEIGH 256
uint32_t* my_texture[TEX_WIDTH*TEX_HEIGH] = {0};

//#define DBG_VERTS
void render(void){    
    if(LOG_RENDER) puts("Render begin...\n");
    
    // world transform (model matrix)
    Matrix44i worldToCamera0 = IDENTITY;
    Matrix44i worldToCamera1 = IDENTITY;

    //angle = M_PI/4.0f;
    worldToCamera0[0][0] = (int32_t)(cos(angle)*(float)ONE);
    worldToCamera0[0][2] = -(int32_t)(sin(angle)*(float)ONE);
    worldToCamera0[2][0] = (int32_t)(sin(angle)*(float)ONE);
    worldToCamera0[2][2] = (int32_t)(cos(angle)*(float)ONE);

    //worldToCamera1[3][0] = FIX_MAKE(2,0); 
    //worldToCamera1[3][1] = 2*ONE;
    worldToCamera1[3][2] = FIX_MAKE(-4,0); 

    if(LOG_RENDER) {
        puts("ANGLE: ");
        put_fixed((int)((180.0f*angle/M_PI) * (float)ONE));
        puts("\n");
    }
    if(angle >= M_PI/2.0f){
        angle = -M_PI/2.0f;
    }
    else{
        angle += M_PI * 0.002f;
    }
    
    // perspective transform (proj matrix)
    Matrix44i Mproj = IDENTITY; 
    int32_t angleOfView = FIX_MAKE(90,0); 
    int32_t near = (ONE/10); // 0.1
    int32_t far = FIX_MAKE(100,0); 
    int32_t imageAspectRatio = DIV(FIX_MAKE(WIDTH, 0), FIX_MAKE(HEIGHT,0));
    int32_t b, t, l, r; 
    gluPerspective(angleOfView, imageAspectRatio, near, far, &b, &t, &l, &r); 
    glFrustum(b, t, l, r, near, far, &Mproj);

#ifdef DBG_VERTS
    puts("ANGLE: 90\n");
    puts("NEAR: ");
    put_fixed(near);
    puts("\nFAR: ");
    put_fixed(far);
    puts("\nASPECT: ");
    put_fixed(imageAspectRatio);

    puts("\nb: ");
    put_fixed(b);
    puts(",t: ");
    put_fixed(t);
    puts(",l: ");
    put_fixed(l);
    puts(",r: ");
    put_fixed(r);
    puts("\n");

    puts("Projection Matrix:\n");
    for (int32_t i = 0; i < 4; i++)
    {
        for (int32_t j = 0; j < 4; j++)
        {
            puts(" ");
            put_fixed(Mproj[i][j]);
        }
        puts("\n");
    }
    
    puts("----------------------------\n");
#endif
    for (int i = 0; i < NVERTS; i+=3)
    {
        Vec3i vert = {g_vertex_buffer_data[i+0], g_vertex_buffer_data[i+1], g_vertex_buffer_data[i+2]};
        Vec3i vertCamera0 = {0, 0, 0};
        Vec3i vertCamera = {0, 0, 0};
        Vec3i projectedVert  = {0, 0, 0};
        multPointMatrix(vert, &vertCamera0, worldToCamera0); 
        multPointMatrix(vertCamera0, &vertCamera, worldToCamera1); 
        multPointMatrix(vertCamera, &projectedVert, Mproj); 
        if (projectedVert.x < -ONE || projectedVert.x > ONE|| projectedVert.y < -ONE || projectedVert.y > ONE) continue; 
    
        // convert to raster space and mark the position of the vertex in the image with a simple dot
        // DISABLED TEST
        //vertices[i/3].x = MIN(WIDTH-1, MAX(0, MUL(WIDTH,  MUL(HALF, ONE + projectedVert.x))));
        //vertices[i/3].y = MIN(HEIGHT-1, MAX(0, MUL(HEIGHT, MUL(HALF, ONE + projectedVert.y))));
        //vertices[i/3].z = MIN(ONE, MAX(-ONE, projectedVert.z));
#ifdef DBG_VERTS
        puts("BUF Vertex[");
        put_digit(i/3);
        puts("]: {");
        put_fixed(vert.x);
        puts(",");
        put_fixed(vert.y);
        puts(",");
        put_fixed(vert.z);
        puts("}\n");

        puts("CAM Vertex[");
        put_digit(i/3);
        puts("]: {");
        put_fixed(vertCamera.x);
        puts(",");
        put_fixed(vertCamera.y);
        puts(",");
        put_fixed(vertCamera.z);
        puts("}\n");

        puts("PRJ Vertex[");
        put_digit(i/3);
        puts("]: {");
        put_fixed(projectedVert.x);
        puts(",");
        put_fixed(projectedVert.y);
        puts(",");
        put_fixed(projectedVert.z);
        puts("}\n");

        puts("SCR Vertex[");
        put_digit(i/3);
        puts("]: {");
        put_digit(vertices[i/3].x);
        puts(",");
        put_digit(vertices[i/3].y);
        puts(",");
        put_fixed(vertices[i/3].z);
        puts("}\n");
#endif
    }
    
    // upload 2d vertices (screen space)
    //gpu_upload_verts(vertices, sizeof(vertices));

    // Reset fence
    g_gpu_fence_var = 0;

    // Model matrix
    cmd_set_uniform_mat4(&g_gpu_driver, UNIFORM_MODEL_MATRIX, (int32_t*)worldToCamera0);
    
    // View matrix
    cmd_set_uniform_mat4(&g_gpu_driver, UNIFORM_VIEW_MATRIX, (int32_t*)worldToCamera1);
    
    // Projection matrix
    cmd_set_uniform_mat4(&g_gpu_driver, UNIFORM_PROJ_MATRIX, (int32_t*)Mproj);
    
    // Time uniform (for animations)
    int32_t time = 0;  // 0 for now...
    cmd_set_uniform_vec4(&g_gpu_driver, UNIFORM_TIME, time, 0, 0, 0);
    
    // Light position
    cmd_set_uniform_vec4(&g_gpu_driver, UNIFORM_LIGHT_POS, 
                         FIX_MAKE(5,0), FIX_MAKE(10,0), FIX_MAKE(5,0), ONE);
    
    // Clear buffers
    cmd_clear(&g_gpu_driver, CLEAR_COLOR_BUFFER | CLEAR_DEPTH_BUFFER, 0xFF000000);

    // Draw
    cmd_draw(&g_gpu_driver, NVERTS/3);

    // ----------------------------------------------------
    // 2. Second cube: farther back + slightly to the left
    // ----------------------------------------------------
    Matrix44i far_cube = IDENTITY;

    // Copy current rotation from your spinning cube
    far_cube[0][0] = worldToCamera0[0][0];
    far_cube[0][2] = worldToCamera0[0][2];
    far_cube[2][0] = worldToCamera0[2][0];
    far_cube[2][2] = worldToCamera0[2][2];

    // Translate it: X = -2.0, Z = -8.0 (much farther back)
    far_cube[3][0] = FIX_MAKE(-2, 0);   // left
    far_cube[3][2] = FIX_MAKE(-8, 0);   // far

    cmd_set_uniform_mat4(&g_gpu_driver, UNIFORM_VIEW_MATRIX, (int32_t*)far_cube);
    cmd_draw(&g_gpu_driver, NVERTS/3);   // <-- second cube
    // (keep the rest: fence, commit, wait – unchanged)
    
    // Fence
    cmd_fence(&g_gpu_driver, (uint32_t)&g_gpu_fence_var, 1);
    
    // Commit
    gpu_commit(&g_gpu_driver);
    
    // Wait
    gpu_wait_fence(&g_gpu_fence_var, 1);

    //delay
    delay(10);
    
    if(LOG_RENDER) puts("Render end...\n");
}

extern void acquire(volatile int32_t* lock);
extern void release(volatile int32_t* lock);

static volatile int32_t lock;

static inline uint32_t r_mhartid(void) {
    uint32_t hartid;
    __asm__ volatile ("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
}

void print_coreid(){
    puts("[Core ");
    put_digit(r_mhartid());
    puts("]\n");
}

#define reg_t uint32_t // RISCV32: register is 32bits

// Saved registers for kernel context switches.
struct context {
    reg_t ra;
    reg_t sp;
  
    // callee-saved
    reg_t s0;
    reg_t s1;
    reg_t s2;
    reg_t s3;
    reg_t s4;
    reg_t s5;
    reg_t s6;
    reg_t s7;
    reg_t s8;
    reg_t s9;
    reg_t s10;
    reg_t s11;
};

extern void sys_switch(struct context *ctx_old, struct context *ctx_new);

#define MAX_TASK 10
#define STACK_SIZE 1024

// global task state
uint8_t task_stack[MAX_TASK][STACK_SIZE];
struct context ctx_tasks[MAX_TASK];
int taskTop=0;  // total number of task

#define NCPU 4
struct cpu {
    struct context ctx_os;
    struct context *ctx_now;
    int current_task;
};

struct cpu cpus[NCPU];
#define CPU (&cpus[r_mhartid()])

// create a new task
int task_create(void (*task)(void))
{
	int i=taskTop++;
	ctx_tasks[i].ra = (reg_t) task;
	ctx_tasks[i].sp = (reg_t) &task_stack[i][STACK_SIZE-1];
	return i;
}

// switch to task[i]
void task_go(int i) {
	CPU->ctx_now = &ctx_tasks[i];
	sys_switch(&CPU->ctx_os, &ctx_tasks[i]);
}

// switch back to os
void task_os() {
	struct context *ctx = CPU->ctx_now;
	CPU->ctx_now = &CPU->ctx_os;
	sys_switch(ctx, &CPU->ctx_os);
}

void os_kernel() {
	task_os();
}

void user_task0(void)
{
    print_coreid();
	puts(" Task0: Created!\n");
	puts("Task0: Now, return to kernel mode\n");
	os_kernel();
	while (1) {
        print_coreid();
		puts(" Task0: Running...\n");
		delay(1000);
		os_kernel();
	}
}

void user_task1(void)
{
    print_coreid();
	puts(" Task1: Created!\n");
	puts("Task1: Now, return to kernel mode\n");
	os_kernel();
	while (1) {
        print_coreid();
		puts(" Task1: Running...\n");
		delay(1000);
		os_kernel();
	}
}

void user_task2(void)
{
    print_coreid();
	puts(" Task2: Created!\n");
	puts("Task2: Now, return to kernel mode\n");
	os_kernel();
	while (1) {
        print_coreid();
		puts(" Task2: Running...\n");
		delay(1000);
		os_kernel();
	}
}

void user_task3(void)
{
    print_coreid();
    puts(" Task3: Created!\n");
	puts("Task3: Now, return to kernel mode\n");
	os_kernel();
	while (1) {
        print_coreid();
		puts(" Task3: Running...\n");
		delay(1000);
		os_kernel();
	}
}

void user_init() {
    print_coreid();
	task_create(&user_task0);
	task_create(&user_task1);
    task_create(&user_task2);
    task_create(&user_task3);
}

void os_start() {
	puts("OS start\n");
	user_init();
}

int os_main(){
    if (r_mhartid() == 0) {
        os_start(); // only hart 0 creates tasks
    }

    while (1) {
        acquire(&lock);
        int t = CPU->current_task;
        if (t < taskTop) {
            task_go(t);
            CPU->current_task = (t + 1) % taskTop;
        }
        release(&lock);
    }
}

// VERTEX SHADER
/*
attribute vec4 vertexPosition;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main() {
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vertexPosition;
}
*/

// FRAGMENT SHADER
/*
void main() {
  gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
*/

int main(){
    // Init only CPU0
    if (r_mhartid() != 0) {
        while(1){

        }
    }

    //fb0 = VGA_SCREEN_0_BASE;
    //fb1 = VGA_SCREEN_1_BASE;

    //return os_main();

    // Initialize GPU Driver
    gpu_init(&g_gpu_driver, g_gpu_ring_ram, GPU_RING_SIZE);

    // VERTEX SHADER (Placeholder)
    static uint32_t vertex_shader[] = {
        0x00000001,
        0x00000002
    };
    
    // FRAGMENT SHADER (Placeholder)
    static uint32_t fragment_shader[] = {
        0x00000003,
        0x00000004
    };
    
    // Set VS
    cmd_set_vertex_shader(&g_gpu_driver, (uint32_t)vertex_shader, sizeof(vertex_shader));

    // Set FS
    cmd_set_fragment_shader(&g_gpu_driver, (uint32_t)fragment_shader, sizeof(fragment_shader));

    // Set VBO
    cmd_set_vbo(&g_gpu_driver, (uint32_t)&g_vertex_buffer_data, 3*sizeof(g_vertex_buffer_data[0]), VBO_FORMAT_FIXED16_16_3D);

    // Set texture
    cmd_set_texture(&g_gpu_driver, 0, (uint32_t)my_texture, TEX_WIDTH, TEX_WIDTH);

    // Commit
    gpu_commit(&g_gpu_driver);

    while(1){
        render();
        /*
        delay(100);
        acquire(&lock);
        delay(100);
        puts("[Core ");
        put_digit(r_mhartid());
        puts("]\n");
        delay(100);
        release(&lock);
        delay(100);
        */
    }
    
    return 0;
}