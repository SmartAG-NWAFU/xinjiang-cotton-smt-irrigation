import pandas as pd

class InitVarietyParameters:
    def __init__(self, id, variety,EmergenceCD, YldFormCD,
                  SenescenceCD, MaturityCD, PlantPop, CGC, CDC, CCx, Zmin,
                   Zmax, Kcb, WP, WPy, HI0, PUP1, PLOW1, PUP2, PUP3):
        self.id = id
        self.variety = variety
        self.EmergenceCD = EmergenceCD
        self.YldFormCD = YldFormCD
        self.SenescenceCD = SenescenceCD
        self.MaturityCD = MaturityCD
        self.PlantPop = PlantPop
        self.CGC = CGC
        self.CDC = CDC
        self.CCx = CCx
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.Kcb = Kcb
        self.WP = WP
        self.WPy = WPy
        self.HI0 = HI0
        self.PUP1 = PUP1
        self.PLOW1 = PLOW1
        self.PUP2 = PUP2
        self.PUP3 = PUP3

    def output_crop_parameters(self):
        return pd.DataFrame({
            'ID': self.id,
            'varieties': self.variety,
            'PlantPop': self.PlantPop,
            'CGC': self.CGC,
            'CDC': self.CDC,
            'CCx': self.CCx,
            'Zmin': self.Zmin,
            'Zmax': self.Zmax,
            'Kcb': self.Kcb,
            'WP': self.WP,
            'WPy': self.WPy,
            'HI0': self.HI0,
            'PUP1': self.PUP1,
            'PLOW1': self.PLOW1,
            'PUP2': self.PUP2,
            'PUP3': self.PUP3
        }, index=[0])
