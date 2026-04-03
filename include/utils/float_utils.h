float *create_rand_float(size_t size){
    float *array = (float*)malloc(size * sizeof(float));
    for (size_t i=0; i < size; i++){
        array[i] = (float)rand() / RAND_MAX;
    }
    return array;
}
