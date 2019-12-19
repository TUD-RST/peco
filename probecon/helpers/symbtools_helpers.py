import pickle
import symbtools.modeltools as mt

def create_save_model(T, V, qq, Q, R, params, file):
    mod = mt.generate_symbolic_model(T, V, qq, Q, dissipation_function=R, simplify=False)
    # calculate state-space model and partial linearization,
    # where the input are the accelerations of the actuated state variables
    mod.calc_state_eq(simplify=False)
    try:
        mod.calc_coll_part_lin_state_eq(simplify=False)
    except:
        print('Collocated partial linearization could not be calculated.')
    # save model parameters
    mod.params = params

    # save model to file
    with open(file, 'wb') as open_file:
        pickle.dump(mod, open_file)
        print('Model file saved.')
    return mod