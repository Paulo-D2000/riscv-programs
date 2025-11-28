#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

// Riscv-Doom Like VGA
#define WIDTH 320
#define HEIGHT 200
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
static volatile int*          gpuVertexQueue = (int*)         0x64000000u;
static volatile unsigned int* gpuVertexSz =    (unsigned int*)0x64001000u;
static volatile unsigned int* gpuBusy =        (unsigned int*)0x64001004u;
static volatile unsigned int* gpuCommit =      (unsigned int*)0x64001008u;

// Dual FB setup
static unsigned char* fb0;
static unsigned char* fb1;

// Riscv-Doom Like UART
static volatile char* UART_BASE = 0x82000000u;

// print string to UART
void _puts(const char* str){
    while(*str){
        *UART_BASE = *str++;
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
    while (value != 0)
    {
        *(pBuf++) = '0' + (value % 10);
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
    int e[2];
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

// vertex buffer
static vec2 vertices[] = {
    {{80,40}},
    {{120,80}},
    {{40,80}},

    {{40,80}},
    {{120,80}},
    {{80,120}},
    
    {{160,40}},
    {{200,80}},
    {{120,80}},

    {{120,80}},
    {{200,80}},
    {{160,120}},

    {{80,120}},
    {{160,120}},
    {{120,160}},

    {{80,120}},
    {{120,80}},
    {{160,120}}
};

void gpu_upload_verts(void* vertices_array, unsigned int size){
    // wait for gpu not busy
    while(*gpuBusy) {};

    // copy vertices
    memcpy((void*)gpuVertexQueue, vertices_array, size);
    *gpuVertexSz = size / 4;

    // commit buffer
    *gpuCommit = 1;
}

void render(void){    
    puts("Render begin...\n");

    //clear_fb(0xFF000000);

    gpu_upload_verts(vertices, sizeof(vertices));

    /*
    // wait for gpu not busy
    while(*gpuBusy) {};

    // copy vertices
    int sz = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            gpuVertexQueue[sz++] = vertices[i].e[j];
        }
    }
    *gpuVertexSz = sz;

    // commit buffer
    *gpuCommit = 1;
    */

    // write 0 to the uart...
    for (int i = 0; i < 4096; i++)
    {
        *UART_BASE = 0;
    }

    // move vertices
    //for (int i = 0; i < 3; i++)
    //{
    //    vertices[i].e[0]+=1;
    //}
    
    //triangle_fill(vertices[0], vertices[1], vertices[2], 0xFF00FF00);

    //swap_fb();

    puts("Render end...\n");
}

int main(){
    fb0 = VGA_SCREEN_0_BASE;
    fb1 = VGA_SCREEN_1_BASE;

    while(1){
        render();
    }
    
    return 0;
}