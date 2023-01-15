'''
Copyright, 2022, Matt Corsaro, matthew_corsaro@brown.edu
'''

from utils import mj_transforms as mjtf

def loadGraspFile(filename, filepath):
    grasp_label_file = filepath + '/' + filename
    f=open(grasp_label_file, 'r')
    grasp_poses = []
    lines = f.readlines()
    for line in lines:
        line = line.replace('[','').replace(']','').replace(',','')
        split_line = line.split()
        if len(split_line) != 7:
            print("Read line that did not contain 7 values:", split_line)
            sys.exit()
        # First 3 - position
        grasp_pos = [float(val) for val in split_line[:3]]
        # Last 4 - quaternion
        grasp_quat = [float(val) for val in split_line[-4:]]
        grasp_pose = mjtf.posRotMat2Mat(grasp_pos, mjtf.quat2Mat(grasp_quat))
        grasp_poses.append(grasp_pose)
    f.close()
    return grasp_poses

def loadGraspQualityFile(filename, filepath):
    grasp_quality_file = filepath + '/' + filename
    f=open(grasp_quality_file, 'r')
    lines = f.readlines()
    grasp_quality_scores = [float(line) for line in lines]
    f.close()
    return grasp_quality_scores

def writeGraspQualityScoreFile(grasp_quality_scores, filename, filepath):
    f = open(filepath + '/' + filename, 'w')
    for score in grasp_quality_scores:
        f.write(str(score))
        f.write('\n')
    f.close()