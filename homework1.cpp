#include <iostream>
#include <windows.h>
#include <stdio.h>
using namespace std;
int main()
{
    char const *p = "lena.bmp";
    int w, h, linebyte;
    unsigned char *buffer;
    FILE *fp = fopen(p, "rb");
    fseek(fp, sizeof(BITMAPFILEHEADER), 0);//重定位指针
    BITMAPINFOHEADER tempbmp;
    fread(&tempbmp, sizeof(BITMAPINFOHEADER), 1, fp); 
    h = tempbmp.biHeight;
    w = tempbmp.biWidth;
    //计算每行的字节数，24：该图片是24位的bmp图，加3确保不丢失像素  
    linebyte = (w * 24 / 8 + 3) / 4 * 4;        
    fseek(fp, 0, SEEK_SET);
    int len = linebyte * h+54;
    buffer = new unsigned char[len];
    fread(buffer, sizeof(char), len, fp);
    FILE *fpr = fopen("./bmp.txt", "w+");

    for (int i = 0; i < len; i++)
        fprintf(fpr, "%6x", *(buffer + i));
    fclose(fpr);

    return 0;
}

