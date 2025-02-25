# operators/_nonparametric.py
"""Classes for operators with no external parameter dependencies."""

__all__ = [
    "create_rom"
]

import torch
import torch.nn as nn

# Set seeds for reproducibility
seed = 2  # You can choose any seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
import opinf.parameters
Params = opinf.parameters.Params()  # Load parameters from dataclass



def create_rom():
    """
    Creates and initializes a Reduced Order Model (ROM) based on the specified parameters.

    Parameters:
    - None

    Returns:
    - torch.nn.Module: The initialized ROM model.
    """
    # Extract flags for each component in the model structure
    has_A = 'A' in Params.model_structure
    has_B = 'B' in Params.model_structure
    has_C = 'C' in Params.model_structure
    has_H = 'H' in Params.model_structure

    # Determine and create the appropriate model based on stability type
    sys_order = Params.ROM_order
    if Params.stability == 'global':
        rom = _ModelHypothesisGlobalStable(
            sys_order=sys_order,
            has_A=has_A,
            has_B=has_B,
            has_C=has_C,
            has_H=has_H
        ).double()
    elif Params.stability == 'local':
        rom = _ModelHypothesisLocalStable(
            sys_order=sys_order,
            has_A=has_A,
            has_B=has_B,
            has_C=has_C,
            has_H=has_H
        ).double()
    else:
        rom = _GeneralModel(
            sys_order=sys_order,
            has_A=has_A,
            has_B=has_B,
            has_C=has_C,
            has_H=has_H
        ).double()

    # Move model to the appropriate device and wrap it for parallel processing
    rom = rom.to(device)
    rom = torch.nn.DataParallel(rom)

    return rom

class _ModelHypothesisGlobalStable(nn.Module):
    """
    Model with Global Stability Guarantees for the ROM.

    Parameters:
    - sys_order (int): System order of the ROM.
    - has_A (bool): Whether to include A operator.
    - has_B (bool): Whether to include B operator.
    - has_C (bool): Whether to include C operator.
    - has_H (bool): Whether to include H operator.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H, *args, **kwargs):
        super().__init__()

        # Initialize model operators based on flags
        operators = []
        print("Global Stability Guarantees!")

        if has_A:
            self.A = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            operators.append('Aq(t)')
        else:
            self.A = None

        if has_B:
            self.B = torch.nn.Parameter(torch.randn(sys_order, Params.input_dim) / 10)
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(torch.randn(sys_order, 1) / 10)
            operators.append('C')
        else:
            self.C = None

        if has_H:
            self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2))
            operators.append('H[q(t) ⊗ q(t)]')
        else:
            print('Warning: Global stability usually requires quadratic matrix')
            self.H = None

        print("ROM structure: dq / dt = " + ' + '.join(operators))
        print(f"Stability: {Params.stability.capitalize()}")

    @property
    def A(self):
        """
        Computes and returns matrix A with global stability guarantees.
        """
        J = self._J - self._J.T
        R = self._R @ self._R.T
        _A = J - R
        self._A = _A
        return self._A

    @property
    def H(self):
        """
        Computes and returns matrix H with global stability guarantees.
        """
        _H_tensor2 = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - _H_tensor2
        self._H = J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)
        return self._H

    def forward(self, x, t, u):
        """
        Computes the ROM output based on input state x, time t, and control input u.

        Parameters:
        - x (torch.Tensor): Input state tensor.
        - t (torch.Tensor): Time tensor.
        - u (torch.Tensor): Control input tensor.

        Returns:
        - torch.Tensor: Computed model output.
        """
        model = torch.zeros_like(x)

        if self.A is not None:
            model += x @ self.A.T

        if self.H is not None:
            model += _kron(x, x) @ self.H.T

        if self.C is not None:
            model += self.C.T

        if self.B is not None and u is not None:
            model += u @ self.B.T

        return model

class _ModelHypothesisLocalStable(nn.Module):
    """
    Model with Local Stability Guarantees for the ROM.

    Parameters:
    - sys_order (int): System order of the ROM.
    - has_A (bool): Whether to include A operator.
    - has_B (bool): Whether to include B operator.
    - has_C (bool): Whether to include C operator.
    - has_H (bool): Whether to include H operator.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H, *args, **kwargs):
        super().__init__()

        # Initialize model operators based on flags
        operators = []
        print("Local Stability Guarantees!")

        if has_A:
            self.A = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            operators.append('Aq(t)')
        else:
            self.A = None

        if has_B:
            self.B = torch.nn.Parameter(torch.randn(sys_order, Params.input_dim) / 10)
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(torch.randn(sys_order, 1) / 10)
            operators.append('C')
        else:
            self.C = None

        if has_H:
            self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2))
            operators.append('H[q(t) ⊗ q(t)]')
        else:
            self.H = None

        print("ROM structure: dq / dt = " + ' + '.join(operators))
        print(f"Stability: {Params.stability.capitalize()}")

    @property
    def A(self):
        """
        Computes and returns matrix A with local stability guarantees.
        """
        J = self._J - self._J.T
        R = self._R @ self._R.T
        Q = self._Q @ self._Q.T
        _A = (J - R) @ Q
        self._A = _A
        return self._A

    def forward(self, x, t, u):
        """
        Computes the ROM output based on input state x, time t, and control input u.

        Parameters:
        - x (torch.Tensor): Input state tensor.
        - t (torch.Tensor): Time tensor.
        - u (torch.Tensor): Control input tensor.

        Returns:
        - torch.Tensor: Computed model output.
        """
        model = torch.zeros_like(x)

        if self.A is not None:
            model += x @ self.A.T

        if self.H is not None:
            model += _kron(x, x) @ self.H.T

        if self.C is not None:
            model += self.C.T

        if self.B is not None and u is not None:
            model += u @ self.B.T

        return model

class _GeneralModel(nn.Module):
    """
    General Model for the ROM without stability guarantees.
    Handles any combination of operators A, B, C, and H.

    Parameters:
    - sys_order (int): System order of the ROM.
    - has_A (bool): Whether to include A operator.
    - has_B (bool): Whether to include B operator.
    - has_C (bool): Whether to include C operator.
    - has_H (bool): Whether to include H operator.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H, *args, **kwargs):
        super().__init__()

        # Initialize model operators based on flags
        operators = []
        if has_A:
            self.A = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            operators.append('Aq(t)')
        else:
            self.A = None

        if has_B:
            self.B = torch.nn.Parameter(torch.randn(sys_order, Params.input_dim) / 10)
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(torch.randn(sys_order, 1) / 10)
            operators.append('C')
        else:
            self.C = None

        if has_H:
            self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2))
            operators.append('H[q(t) ⊗ q(t)]')
        else:
            self.H = None

        print("ROM structure: dq / dt = " + ' + '.join(operators))
        print(f"Stability: {Params.stability.capitalize()}")

    def forward(self, x, t, u):
        """
        Computes the ROM output based on input state x, time t, and control input u.

        Parameters:
        - x (torch.Tensor): Input state tensor.
        - t (torch.Tensor): Time tensor.
        - u (torch.Tensor): Control input tensor.

        Returns:
        - torch.Tensor: Computed model output.
        """
        model = torch.zeros_like(x)

        if self.A is not None:
            model += x @ self.A.T

        if self.H is not None:
            model += _kron(x, x) @ self.H.T

        if self.C is not None:
            model += self.C.T

        if self.B is not None and u is not None:
            model += u @ self.B.T

        return model


def _kron(x, y):
    """
    Compute the Kronecker product of two tensors.

    Parameters:
    - x (torch.Tensor): First tensor.
    - y (torch.Tensor): Second tensor.

    Returns:
    - torch.Tensor: Kronecker product of x and y.
    """
    return torch.einsum("ab,ad->abd",
                        [x, y]).view(x.size(0), x.size(1) * y.size(1))
