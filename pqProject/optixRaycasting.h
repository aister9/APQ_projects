#pragma once


enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

struct Ray;
struct Hit;

struct Params
{
    OptixTraversableHandle handle;
    Ray* rays;
    Hit* hits;
};

