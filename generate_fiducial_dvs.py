import os
import shutil
import subprocess

class Cosmology:
    def __init__(self, name, h=0.67, Omega_b=0.049, Omega_m=0.319, As=2.1e-9, ns=0.96, w=-1, wa=0):
        self.h: float = h
        self.Omega_b: float = Omega_b
        self.Omega_m: float = Omega_m
        self.As: float = As
        self.ns: float = ns
        self.w: float = w
        self.wa: float = wa
        self.name: str = name

    def __repr__(self):
        return f"(h={self.h}, Omega_b={self.Omega_b}, Omega_m={self.Omega_m}, As={self.As}, ns={self.ns}, w={self.w}, wa={self.wa})"

w_low  = -1.15
w_hi   = -0.85
wa_low = -0.35
wa_hi = 0.25
Omega_m_low = 0.28
Omega_m_hi = 0.36
As_low = 1.9e-9
As_hi = 2.3e-9
ns_low = 0.94
ns_hi = 0.98

fiducials = [
    Cosmology("wlow_walow",                                 w=w_low, wa=wa_low),
    Cosmology("whi_walow",                                  w=w_hi,  wa=wa_low),
    Cosmology("wlow_wahi",                                  w=w_low, wa=wa_hi),
    Cosmology("whi_wahi",                                   w=w_hi,  wa=wa_hi),
    Cosmology("Omega_mlow_wlow_walow", Omega_m=Omega_m_low, w=w_low, wa=wa_low),
    Cosmology("Omega_mlow_whi_walow",  Omega_m=Omega_m_low, w=w_hi,  wa=wa_low),
    Cosmology("Omega_mlow_wlow_wahi",  Omega_m=Omega_m_low, w=w_low, wa=wa_hi),
    Cosmology("Omega_mlow_whi_wahi",   Omega_m=Omega_m_low, w=w_hi,  wa=wa_hi),
    Cosmology("Omega_mhi_wlow_walow",  Omega_m=Omega_m_hi,  w=w_low, wa=wa_low),
    Cosmology("Omega_mhi_whi_walow",   Omega_m=Omega_m_hi,  w=w_hi,  wa=wa_low),
    Cosmology("Omega_mhi_wlow_wahi",   Omega_m=Omega_m_hi,  w=w_low, wa=wa_hi),
    Cosmology("Omega_mhi_whi_wahi",    Omega_m=Omega_m_hi,  w=w_hi,  wa=wa_hi),
    Cosmology("Aslow_wlow_walow",      As=As_low,           w=w_low, wa=wa_low),
    Cosmology("Aslow_whi_walow",       As=As_low,           w=w_hi,  wa=wa_low),
    Cosmology("Aslow_wlow_wahi",       As=As_low,           w=w_low, wa=wa_hi),
    Cosmology("Aslow_whi_wahi",        As=As_low,           w=w_hi,  wa=wa_hi),
    Cosmology("Ashi_wlow_walow",       As=As_hi,            w=w_low, wa=wa_low),
    Cosmology("Ashi_whi_walow",        As=As_hi,            w=w_hi,  wa=wa_low),
    Cosmology("Ashi_wlow_wahi",        As=As_hi,            w=w_low, wa=wa_hi),
    Cosmology("Ashi_whi_wahi",         As=As_hi,            w=w_hi,  wa=wa_hi),
    Cosmology("nslow_wlow_walow",      ns=ns_low,           w=w_low, wa=wa_low),
    Cosmology("nslow_whi_walow",       ns=ns_low,           w=w_hi,  wa=wa_low),
    Cosmology("nslow_wlow_wahi",       ns=ns_low,           w=w_low, wa=wa_hi),
    Cosmology("nslow_whi_wahi",        ns=ns_low,           w=w_hi,  wa=wa_hi),
    Cosmology("nshi_wlow_walow",       ns=ns_hi,            w=w_low, wa=wa_low),
    Cosmology("nshi_whi_walow",        ns=ns_hi,            w=w_hi,  wa=wa_low),
    Cosmology("nshi_wlow_wahi",        ns=ns_hi,            w=w_low, wa=wa_hi),
    Cosmology("nshi_whi_wahi",         ns=ns_hi,            w=w_hi,  wa=wa_hi),
]

def generate_yaml(cosmo):
    print(f"[INFO] Generating yaml file for fiducial {cosmo}")
    yaml_path = f"GENERATE_FIDUCIAL_{cosmo.name}.yaml"
    shutil.copyfile("GENERATE_FIDUCIAL_TEMPLATE.yaml", yaml_path)
    with open(yaml_path, "r") as f: contents = f.read()
    contents = contents.replace("%FIDCOSMO%", f"{cosmo.name}")
    contents = contents.replace("%H0%",       f"{100*cosmo.h}")
    contents = contents.replace("%Omega_b%",  f"{cosmo.Omega_b}")
    contents = contents.replace("%Omega_m%",  f"{cosmo.Omega_m}")
    contents = contents.replace("%As1e9%",    f"{cosmo.As*1e9}")
    contents = contents.replace("%ns%",       f"{cosmo.ns}")
    contents = contents.replace("%w%",        f"{cosmo.w}")
    contents = contents.replace("%wa%",       f"{cosmo.wa}")
    contents = contents.replace("%w0pwa%",    f"{cosmo.w+cosmo.wa}")
    with open(yaml_path, "w") as f: f.write(contents)
    return yaml_path

def run_yaml(yaml_path):
    print(f"[INFO] Running cobaya on yaml {yaml_path}")
    os.chdir("../../")
    p = subprocess.run(["cobaya-run", f"./projects/lsst_y1/{yaml_path}", "-f"])
    if p.returncode != 0: 
        print(f"[ERROR] Failed to run cobaya on yaml {yaml_path}")
        cleanup(yaml_path)
    else: print("[INFO] Generated successfully")
    os.chdir("./projects/lsst_y1")

def cleanup(yaml_path):
    print(f"[INFO] Removing yaml {yaml_path}")
    os.remove(yaml_path)

for cosmo in fiducials:
    yaml_path = generate_yaml(cosmo)
    run_yaml(yaml_path)
    cleanup(yaml_path)