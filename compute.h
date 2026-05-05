#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#ifdef __cplusplus
extern "C" {
#endif

void initDeviceMemory();
void freeDeviceMemory();
void copyDeviceToHost();
void compute();

#ifdef __cplusplus
}
#endif

#endif
