from argparse import Action
import os
import time

def main():
    frames = [2,4,8,10,14,18,22,25]
    actions = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
    actions = ["walking", "eating", "smoking", "directions","greeting", 
                    "phoning", "purchases", "takingphoto", "walkingdog","walkingtogether"]
    training_start = time.time()
    for action_idx in range(2,10): 
        for i in range(0,8):
            for j in range(0, 3):
                
                training_path = "cmd /c python main_h36_ang.py --input_n 10 --output_n {} --skip_rate 1 --joints_to_consider 16 --action {} --model_path ./checkpoints/CKPT_3D_H36M/h36_ang_{}frames_ckpt".format(frames[i], actions[action_idx], frames[i])
                testing_path = "cmd /c python main_h36_ang.py --input_n 10 --output_n {} --skip_rate 1 --joints_to_consider 16 --action {} --mode test --model_path ./checkpoints/CKPT_3D_H36M/h36_ang_{}frames_ckpt".format(frames[i], actions[action_idx], frames[i])
                
                results = open("results_{}.txt".format(actions[action_idx]), 'a')
                if not (i==0 and j==0): 
                    results.write("Training time: {}".format(end-start))
                results.write("\n\nPrediction of {} Frames\n".format(frames[i]))
                results.write("Iteration {}\n".format(j+1))
                results.close()

                print("--------------------------Recording {}--------------------------".format(j+1))
                print(">>> Automating Prediction of {} Frames - Training Process started...".format(frames[i]))
                start = time.time()
                os.system(training_path)
                end = time.time()
                print(">>> Automating Prediction of {} Frames - Testing Process started...".format(frames[i]))
                os.system(testing_path )
            print(">>> Automating Process Completed for {} Frames\n\n\n".format(frames[i]))    
        training_end = time.time()
        print("Total time to train 'walking' action: {}".format(training_end - training_start))


if __name__=='__main__':
    main()