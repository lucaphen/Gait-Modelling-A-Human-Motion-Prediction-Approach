from argparse import Action
import os
import time

def main():
    # actions = ["walking", "eating", "smoking", "discussion", "directions",
    #                 "greeting", "phoning", "posing", "purchases", "sitting",
    #                 "sittingdown", "takingphoto", "waiting", "walkingdog",
    #                 "walkingtogether"]
    actions = ["walking", "eating", "smoking", "directions","greeting", 
                    "phoning", "purchases", "takingphoto", "walkingdog","walkingtogether"]
    models = ["residual_unsup_SA","untied"]
    
    training_start = time.time()
    for model in models: 
        for action_idx in range(0,10): 
            for iteration in range(0,3):
                if model == "residual_unsup_SA":
                    cmd = "cmd /c python src/translate.py --residual_velocities --action {} --model residual_unsup_SA --iterations 10000  --seq_length_in 10 --seq_length_out 25 --learning_rate 0.005 --omit_one_hot".format(actions[action_idx])
                elif model == "untied":
                    cmd = "cmd /c python src/translate.py --residual_velocities --action {} --model untied --iterations 10000 --seq_length_in 10 --seq_length_out 25 --learning_rate 0.005 --architecture basic".format(actions[action_idx])

                results = open("results_{0}_{1}.txt".format(model, actions[action_idx]), 'a')
                if not (iteration==0 and action_idx==0): 
                    results.write("Training time: {}".format(end-start))
                results.write("\n\nHumanMotionPrediction Estimates of {}\n".format(model))
                results.write("Iteration {}\n".format(iteration+1))
                results.close()
                
                print("--------------------------Recording {}--------------------------".format(iteration+1))
                print(">>> Automating Prediction of {} - Training Process started...".format(model))
                start = time.time()
                os.system(cmd)
                end = time.time()
        training_end = time.time()
        print("Total time to train action: {}".format(training_end - training_start))


if __name__=='__main__':
    main()