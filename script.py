import allqueries
import load_data
import numpy as np

cam_numbers = [120]#np.linspace(30,130,11)
exp_results = []
for i in cam_numbers:
    cam_n = int(i)
    load_data.run(cam_n)
    results = np.zeros(7)
    for j in range(1):
        cost1,recall1,precision1= allqueries.run(cam_n,0)
        cost2,recall2,precision2,delay= allqueries.run(cam_n,1,0)
        results[0] += cost1
        results[1] += recall1
        results[2] += precision1
        results[3] += cost2
        results[4] += recall2
        results[5] += precision2
        results[6] += delay

    results = results/1
    exp_results.append((i,results[0],results[1],results[2],results[3],results[4],results[5]))
    print("=> Camera number : ",i)
    print("  model   | # cost | # recall | # precision")
    print("  ------------------------------")
    print("  basel    | {:8f} | {:8f} | {:8f} |".format(results[0],results[1],results[2]))
    print("  rexcamo  | {:8f} | {:8f} | {:8f} |".format(results[3],results[4],results[5]))
    print("  delay | {:8f} |".format(results[6]))
    print("  ------------------------------")
    #np.save("beijing_result_80.npy",np.array(exp_results))


