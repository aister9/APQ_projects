# APQ_projects


OptixTutorial copyright by Ingo Wald.     
The origin code is here: https://github.com/ingowald/optix7course



For the optix ptx file, I use gen_ptx.bat.  
That call nvcc and compile devicePrograms.cu to deviceProgams.ptx.   
The code:   
~~~
nvcc.exe --machine=64 --ptx --gpu-architecture=compute_52 --use_fast_math --relocatable-device-code=true --generate-line-info -Wno-deprecated-gpu-targets devicePrograms.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" -o devicePrograms.ptx -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.4.0\include" -I"common\gdt"
~~~
Need changes the --ccbin and --I to your **VS path** and **OptiX path**.
