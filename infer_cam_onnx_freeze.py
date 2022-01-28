import onnxruntime as ort
import numpy as np
import cv2

import time

def gen_frame(path_video,w,h):
    while True:
        cap=cv2.VideoCapture(path_video)
        print("open")
        while True:
            flag,frame=cap.read()
            if flag:
                yield cv2.resize(frame,(w,h)).astype("float32")#/255.
                continue
            else:
                cap.release()
                break

def main():
    sess = ort.InferenceSession('freeze_res50_sim.onnx')
    # sess = ort.InferenceSession('freeze_mbv3_sim.onnx')

    io = sess.io_binding()


    # #for mobilenet
    # rec = [ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 16, 90, 160], dtype=np.float32), 'cuda'),\
    #         ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 20, 45, 80], dtype=np.float32), 'cuda'),\
    #         ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 40, 23, 40], dtype=np.float32), 'cuda'),\
    #         ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 64, 12, 20], dtype=np.float32), 'cuda')]


    #for resnet50
    rec = [ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 16, 90, 160], dtype=np.float32), 'cuda'),\
            ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 32, 45, 80], dtype=np.float32), 'cuda'),\
            ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 64, 23, 40], dtype=np.float32), 'cuda'),\
            ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 128, 12, 20], dtype=np.float32), 'cuda')]


    # Set output binding.
    for name in ['com', 'r1o', 'r2o', 'r3o', 'r4o']:
        io.bind_output(name, 'cuda')


    cap=cv2.VideoCapture(0)


    w=1280
    h=720

    cap.set(3, w)  # width=1280
    cap.set(4, h)  # height=720

    flag,frame_init=cap.read()





    bak_bgr=np.ones_like(frame_init)
    bak_bgr*=np.asarray([120, 255, 155],dtype="uint8")
    bak_bgr=bak_bgr.astype("float32")#/255.

    baks={}
    for k,i in zip([49,50,51,52],[1,2,3,4]):
        baks[k]=cv2.resize(cv2.imread("backgrounds/bak{}.jpg".format(i)),(w,h)).astype("float32")#/255.

    gen=gen_frame("backgrounds/inception_10s.mp4",w,h)

    baks[48]=bak_bgr
    baks[53]=gen

    # Inference loop
    k=-1
    while not k==27:
        _,frame=cap.read()

        start = time.time()


        src=frame[None,:].astype("float32")#/255.
        bak=bak_bgr[None,:]

        
        io.bind_cpu_input('src', src)
        io.bind_cpu_input('bak', bak)
        io.bind_ortvalue_input('r1i', rec[0])
        io.bind_ortvalue_input('r2i', rec[1])
        io.bind_ortvalue_input('r3i', rec[2])
        io.bind_ortvalue_input('r4i', rec[3])

        sess.run_with_iobinding(io)

        com, *rec = io.get_outputs()
        com=com.numpy()[0]


        if k>0:
            print(k)
            bak=baks.get(k,bak_bgr)
            if not isinstance(bak,np.ndarray):
                bak_bgr=next(bak)
            else:
                bak_bgr=bak

        if not k==ord("q"):
            cv2.imshow("com",com)
        else:
            print(k)
            cv2.imshow("com",frame)

        end = time.time()
        fps  = 1 / (end-start)
        print( "Estimated frames per second : {0}".format(fps))

        k_n=cv2.waitKey(1)
        k=(k if k_n<0 else k_n)
    cap.release()

if __name__ == "__main__":
    # import line_profiler
    # profile = line_profiler.LineProfiler(main)  # 把函数传递到性能分析器
    # profile.enable()  # 开始分析
    main()
    # profile.disable()  # 停止分析
    # profile.print_stats()
