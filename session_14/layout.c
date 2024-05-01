#include <stdio.h>

typedef struct {
    float x;
    float y;
    float w;
    float h;
} NFVT_Layout_Rect;

typedef enum {
    NFVT_LO_HORZ,
    NFVT_LO_VERT,
} NFVT_Layout_Orient;

typedef struct {
    NFVT_Layout_Rect rect;
    NFVT_Layout_Orient orient;
    size_t count;
} NFVT_Layout;

void nfvt_widget(NFVT_Layout_Rect r)
{

}

NFVT_Layout_Rect nfvt_layout_slot(NFVT_Layout *l, size_t i)
{
    NFVT_Layout_Rect r = {0};
    switch (l->orient) {
        case NFVT_LO_HORZ:
            break;
        case NFVT_LO_VERT:
            break;
}

int main(void)
{
    size_t width = 1920;
    size_t height = 1080;

    NFVT_Layout root = {
        .orient = NFVT_LO_HORZ,
        .rect = {0, 0, width, height},
        .count = 3,
    };

    return 0;
}
