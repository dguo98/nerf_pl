import numpy as np
import os
import sys

exp_id = 44

def generate(gpu, epoch, bs, lr, img_ds, model, box_warp, n_feat, back_res, use_xyz_net, n_xyz_dim,mode):
    global exp_id
    exp_id += 1
    exp_name = "T%05d" % exp_id
    print("exp_name=",exp_name)

    f = open("template_noat.sh")
    lines = f.readlines()
    f.close()

    lines[0] = f"EXP_NAME={exp_name}\n"
    lines[8] = f"GPU={gpu}\n"
    lines[9] = f"BOX_WARP={box_warp:.2f}\n"
    lines[10] = f"IMG_DS={img_ds}\n"
    lines[11] = f"MODEL={model}\n"
    lines[12] = f"N_FEAT={n_feat}\n"
    lines[13] = f"BACKBONE_RES={back_res}\n"
    lines[14] = f"BS={bs}\n"
    lines[15] = f"LR={lr:.7f}\n"
    lines[16] = f"EPOCHS={epoch}\n"
    lines[17] = f"USE_XYZ_NET={use_xyz_net}\n"
    lines[18] = f"N_XYZ_DIM={n_xyz_dim}\n"
    lines[19] = f"MODE={mode}\n"

    f = open("%s.sh" % exp_name, "w")
    for line in lines:
        f.write(line)
    f.close()
    os.system("chmod +x %s.sh" % exp_name)
    return "%s.sh" % exp_name
    
if __name__ == "__main__":

    epoch=20
    bs=1024
    lr=5e-4
    img_ds=8
    model="NeRFTriplane"
    box_warp=1.0
    n_feat=48
    back_res=512
    use_xyz_net=1
    n_xyz_dim=128
    mode="debug"
    
    # use gpu=1,2,3,4,5,6
    
    ind = 0
    scripts = {}
    
    for lr in [1e-3, 1e-4, 5e-4]:
        for model in ["NeRF", "NeRFCube", "NeRFTriplane"]:
            if model == "NeRFTriplane":
                back_res=512
            else:
                back_res=128
            box_warp = 2.0
            mode="default"

            gpu=(ind%3)+5
            ind+=1
            script = generate(gpu, epoch, bs, lr, img_ds, model, box_warp, n_feat, back_res, use_xyz_net, n_xyz_dim, mode)
            if not gpu in scripts:
                scripts[gpu] = []
            scripts[gpu].append(script)

   
    for gpu in scripts:
        f = open(f"sweep_gpu{gpu}.sh", "w")
        cmd = ";".join(["./" + s for s in scripts[gpu]] )
        f.write(cmd + "\n")
        f.close()
        os.system(f"chmod +x sweep_gpu{gpu}.sh")
