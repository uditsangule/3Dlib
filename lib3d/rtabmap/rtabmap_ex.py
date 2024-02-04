import os
path = os.path.dirname(__file__)
def export(inputdb , filex='_',ascii=True , ba=False, decimation=2 ,scan=False, optim=True , poses=True,mesh=False , formatpose=11):
    stat = 0
    laser = f'--decimation {decimation}' if decimation > 0 else ''
    return os.system(f"{path}/rtabmap-export {'--custom_optimization' if optim else ''} {'--ba' if ba else ''} \
                      --images_id --ground_normals_up {laser} {'--scan' if scan else ''} {'--mesh' if mesh else ''} --prop_radius_factor 0.05 --poses --poses_format {formatpose} --outputdir {os.path.dirname(inputdb) + os.sep} --output {filex} {'--ascii' if ascii else ''} {inputdb}")

def detectloopclosure(inputdb , iter=12, rmax=3 , rmin=0 ):
    return os.system(f"{path}/rtabmap-detectMoreLoopClosures -rx {rmin} -r {rmax} -i {iter} --intra {inputdb}")

def globaloptimization(inputdb , strategy=2 , iter=12):
    return os.system(f"{path}/rtabmap-globalBundleAdjustment --Optimizer/Strategy {strategy} --Optimizer/Iterations {iter} {inputdb}")
