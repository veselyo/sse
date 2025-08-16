import numpy as np
import torch
from complete_algorithm import sse_search_algorithm, Box


# Globals
MESH_DEFAULT_PARAMS = [29.0438, 36.2985, -6.27078, -30.9564, 7.54789,
                       14.0911, 16.1414, -44.3358, -10.4607, 96.0738,
                       -27.4556, 9.05684, -8.51091, 26.926, -1.84734, 0.625359]

MIN_LENGTH_MM = -150
MAX_LENGTH_MM = 150
MIN_OVERHANG_MM = -150
MAX_OVERHANG_MM = 100
MIN_WIDTH_MM = -100
MAX_WIDTH_MM = 100
MIN_GREENHOUSE_TAPER_MM = -100
MAX_GREENHOUSE_TAPER_MM = 100

CD_TARGET = 0.27


class NeuralNetwork(torch.nn.Module):
    """
    Defines architecture of neural network for drag coefficient prediction
    so that it can be loaded from file.
    """
    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(16, 128)
        self.drop1 = torch.nn.Dropout(p=0.2, inplace=False)
        self.act1 = torch.nn.SELU()
        self.hidden2 = torch.nn.Linear(128, 128)
        self.drop2 = torch.nn.Dropout(p=0.2, inplace=False)
        self.act2 = torch.nn.SELU()
        self.output = torch.nn.Linear(128, 1)
        self.act_output = torch.nn.ReLU()

    def forward(self, x):
        x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
        x = self.act1(self.hidden1(x))
        x = self.drop1(x)
        x = self.act2(self.hidden2(x))
        x = self.drop2(x)
        x = self.act_output(self.output(x))
        return x


def performance_function(args: np.ndarray) -> float:
    """
    Performance function for AERIS.
    
    Input
    -----
    args : np.ndarray of shape (4,)
           [length  overhang  width  greenhouse] offsets in mm
              
    Returns
    -------
    perfromance_val  :  Good designs satisfy performance_val <= 0. 
    """
    # Edge cases
    if args is None:
        raise ValueError("No arguments found.")
    if not isinstance(args, np.ndarray):
        raise TypeError("Args must be np.ndarray.")
    if args.shape != (4,):
        raise ValueError("Args must be of length 4.")

    l, o, w, g = args

    # Constraint: Input must be in the defined min and max intervals
    if (l < MIN_LENGTH_MM or l > MAX_LENGTH_MM) or (
        o < MIN_OVERHANG_MM or o > MAX_OVERHANG_MM) or (
        w < MIN_WIDTH_MM or w > MAX_WIDTH_MM) or (
        g < MIN_GREENHOUSE_TAPER_MM or g > MAX_GREENHOUSE_TAPER_MM):
        return 1
    
    # Build input for NN
    params = MESH_DEFAULT_PARAMS.copy()
    params[0] = l
    params[3] = o
    params[1] = w
    params[8] = g
    nn_input = torch.tensor(np.array(params,dtype=np.float32),
                            dtype=torch.float32)
    nn_input.requires_grad = True

    # Predict Cd
    Cd_nn_output = CD_MODEL(nn_input)
    Cd = float(Cd_nn_output.to(torch.float32).detach().numpy()[0])
    
    # Compute performance val
    perfromance_val = (Cd - CD_TARGET) / CD_TARGET
    return perfromance_val


def performance_criterion(performance_val: float) -> bool:
    """
    Return True when the given performance value satisfies the performance
    criterion performance_function(args, Cd_target) <= 0.
    """
    if performance_val is None:
        raise ValueError("No performance value found.")
    if not isinstance(performance_val, float):
        raise TypeError("Performance value must be a float.")
    return performance_val <= 0.0


if __name__ == "__main__":
    # Define design space and one good design (optimal point)
    lower = np.array([MIN_LENGTH_MM,
                      MIN_OVERHANG_MM,
                      MIN_WIDTH_MM,
                      MIN_GREENHOUSE_TAPER_MM], dtype=float)
    upper = np.array([MAX_LENGTH_MM,
                      MAX_OVERHANG_MM,
                      MAX_WIDTH_MM,
                      MAX_GREENHOUSE_TAPER_MM], dtype=float)
    design_space = Box(lower, upper)
    x_0 = np.array([120, -120, 55, -25], dtype=float)

    # Load trained model for predicting drag coefficient
    CD_MODEL = NeuralNetwork()
    CD_MODEL.load_state_dict(torch.load("aeris/dnn_model_torch.pth",
                                        weights_only=True,
                                        map_location=torch.device("cpu")))
    CD_MODEL.eval()

    # Run SSE
    sse_box = sse_search_algorithm(
        f=performance_function,
        performance_criterion=performance_criterion,
        design_space=design_space,
        x_0=x_0,
        N1=100,
        N2=700,
        growth_rate=0.05,
        confidence_lower=0.99,
        confidence_upper=1.00,
        alpha_c=0.001,
        change_threshold=0.01,
        params_names=["length", "overhang", "width", "greenhouse"]
    )