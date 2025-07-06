from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.Execution.BasicRunner import BasicRunner
import os

class OpenFOAMParallelRunner:
    def __init__(self, case_dir, n_proc=6):
        self.case_dir = case_dir
        self.n_proc = n_proc
        
    def run_blockMesh(self):
        runner = UtilityRunner(argv=["blockMesh", "-case", self.case_dir], silent=False)
        runner.start()
        runner.wait()
        return runner.run.returncode == 0
    
    def run_surfaceFeatureExtract(self, dictPath):
        runner = UtilityRunner(argv=["surfaceFeatureExtract", "-case", self.case_dir, "-dict", dictPath], silent=False)
        runner.start()
    
    def run_snappyHexMesh(self):
        runner = UtilityRunner(argv=["snappyHexMesh", "-overwrite", "-case", self.case_dir], silent=False)
        runner.start()

    def decompose_case(self):
        runner = UtilityRunner(argv=["decomposePar", "-force", "-case", self.case_dir], silent=False)
        runner.start()

    def run_parallel_simpleFoam(self):
        mpirun_cmd = ["mpirun", "-np", str(self.n_proc), "simpleFoam", "-case", self.case_dir, "-parallel"]
        runner = BasicRunner(argv=mpirun_cmd, silent=False)
        runner.start()

    def reconstruct_case(self):
        runner = UtilityRunner(argv=["reconstructPar", "-case", self.case_dir], silent=False)
        runner.start()
    
    def run_all_surfaceFeatureExtract(self):
        dicts = [
            "system/surfaceFeatureExtract_mainBodyDict",
            "system/surfaceFeatureExtract_FLDict",
            "system/surfaceFeatureExtract_FRDict",
            "system/surfaceFeatureExtract_RLDict",
            "system/surfaceFeatureExtract_RRDict",
        ]
        for dict_path in dicts:
            if not self.run_surfaceFeatureExtract(dict_path):
                return False
        return True
    
    def run_all(self):
        if not self.run_blockMesh():
            return False
        if not self.run_all_surfaceFeatureExtract():
            return False
        if not self.run_snappyHexMesh():
            return False
        if not self.decompose_case():
            return False
        if not self.run_parallel_simpleFoam():
            return False
        if not self.reconstruct_case():
            return False
        return True

if __name__ == "__main__":
    case = "../case"
    runner = OpenFOAMParallelRunner(case_dir=case, n_proc=6)
    runner.run_all()
    print("Parallel run completed successfully")
