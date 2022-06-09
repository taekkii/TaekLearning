

import time

last_timestamp = None
debug_mode = False
overhead_dict = {}
wait_iter = 0


def stamp(caption:str="" , verbose=False):
    global last_timestamp, debug_mode , overhead_dict
    if not debug_mode: return


    if last_timestamp is None  or  caption=="":
        last_timestamp = time.time()
        return
    if caption not in overhead_dict:
        overhead_dict[caption] = 0.0
        
    delta = time.time()-last_timestamp
    overhead_dict[caption]+=delta

    if verbose: print(f"[TIMESTAMP] ({caption}) {delta:.3f}s")

    last_timestamp = time.time()

def prt(*msg):
    global last_timestamp,debug_mode
    if not debug_mode: return
    t0=time.time()
    
    print("[DEBUG PRINT]",end='')
    print(*msg)
    
    last_timestamp += time.time()-t0

def summary():
    global debug_mode
    if not debug_mode: return

    sum_t = sum(overhead_dict.values())
    
    print("========== [OVERHEAD SUMMARY] ==========")
    for caption, t in overhead_dict.items():
        print(f"({caption:^20}) --> {t*100.0/sum_t:.2f}%")

    print()
    input()


def wait(cycle=1):
    global debug_mode,last_timestamp,wait_iter
    if not debug_mode: return
    
    wait_iter+=1
    if wait_iter % cycle !=0:return

    t0=time.time()
    summary()
    
    last_timestamp += time.time()-t0

def on():
    global debug_mode
    debug_mode=True
    stamp()

def off():
    global debug_mode
    debug_mode=False
