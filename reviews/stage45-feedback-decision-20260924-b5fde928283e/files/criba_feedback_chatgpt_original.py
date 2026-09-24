"""Criba cinemática y de coste; sólo biblioteca estándar, sin simulación.
Usa geometría/tiempos del paquete 7186929. No estima respuesta neuronal.
Los costes son escalados linealmente desde corridas de 400 ms con preparación.
"""
import argparse, hashlib, json, math
from pathlib import Path
GEOM = "3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3"
FIELDS = "fcf9f2879d703cc03e3b8dca3908701279f12ce113188ce1eaf8c65484401133"
def need(ok, msg):
    if not ok: raise ValueError(msg)
def read(p, hashes, expected=None):
    b=p.read_bytes(); h=hashlib.sha256(b).hexdigest()
    need(expected is None or h==expected, "Hash geométrico distinto: "+str(p))
    hashes[str(p)]=h
    return json.loads(b)
def run(folder, T, tp, angle):
    need(all(math.isfinite(x) for x in (T,tp,angle)) and 0<=tp<T and 0<angle<=180,
         "Se requiere 0<=pulso<horizonte y 0<ángulo<=180")
    hashes={}
    g=read(folder/"runs/plus/executed_sources/16__etapa4_diseno_20260923_17__GEOMETRY.json",hashes,GEOM)
    fields=read(folder/"recovery/CAMPOS.json",hashes,FIELDS)
    close=read(folder/"recovery/CLOSE_01.json",hashes)
    old=read(folder/"prior/CLOSE_01.json",hashes)
    n400=(old["budget"]["aggregate_wall_s"]-old["reference_plus"]["wall_s"])/2
    r400=sum(close["runs"][s]["wall_s"] for s in ("plus","minus"))/2
    need(n400>0 and r400>0, "Tiempos no positivos")
    q=g["prepared_qpos_root"]; a=g["prepared_antennae_mm"]
    scale=g["body_native_units_per_mm"]; v=g["forward_command_mm_s"]
    need(scale>0 and v>0 and len(q)==7 and len(a)==2, "Geometría inválida")
    origin=[x/scale for x in q[:2]]
    w,x,y,z=q[3:]; yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    rest=T-tp; alpha=math.radians(angle); out={}
    for side,f in fields.items():
        need(f["geometry_sha256"]==GEOM, "Fuentes ligadas a otra geometría")
        source=f["source_mm"];sigma=f["sigma_mm"]
        beta=math.atan2(source[1]-origin[1],source[0]-origin[0])-yaw
        beta=math.atan2(math.sin(beta),math.cos(beta))
        # Rotación rígida contrafactual que aleja el rumbo inicial de la fuente.
        turn=-math.copysign(alpha,beta);ca,sa=math.cos(turn),math.sin(turn)
        def concentrations(rotated):
            result=[]
            for p in a:
                dx,dy=p[0]-origin[0],p[1]-origin[1]
                xy=[origin[0]+ca*dx-sa*dy,origin[1]+sa*dx+ca*dy] if rotated else p[:2]
                result.append(math.exp(-math.dist(xy,source)**2/(2*sigma*sigma)))
            return result
        c0,c1=concentrations(False),concentrations(True)
        out[side]={"bearing_initial_deg":math.degrees(beta),
            "distance_initial_mm":math.dist(origin,source),
            "straight_distance_over_v_s":math.dist(origin,source)/v,
            "counterfactual_rotation_deg":math.degrees(turn),
            "c_initial":c0,"c_after_rigid_rotation_at_initial_pose":c1,
            "delta_normalized_LR":(c1[0]-c1[1])/sum(c1)-(c0[0]-c0[1])/sum(c0)}
    n,r=n400*T/.4/3600,r400*T/.4/3600
    return {"inputs_sha256":hashes,"T_s":T,"perturbation_s":tp,
        "remaining_s":rest,"assumed_decoder_limit_deg_s":5.0,
        "maximum_command_integral_after_perturbation_deg":5*rest,
        "command_time_for_angle_at_ceiling_s":angle/5,
        "earliest_command_only_compensation_from_on_s":tp+angle/5,
        "geometry":out,
        "ideal_equal_speed_position_separation_upper_mm":2*v*rest*math.sin(alpha/2),
        "cost_linear_h":{"one_native":n,"one_reference":r,
            "donor_and_two_native_and_two_reference":3*n+2*r,
            "three_native_and_three_reference":3*(n+r)},
        "limits":["Geometría en el preparado, NO en el instante futuro del pulso.",
            "La integral de mando no acota toda rotación física pasiva.",
            "Cota XY sólo para dos trayectorias ideales de igual rapidez y diferencia de rumbo <=ángulo.",
            "Tiempo extrapolado, no corrida larga medida ni presupuesto autorizado.",
            "No efectos neurales, contactos, navegación o etapa5 simulados."],
        "organism_executed":False,"CUDA_executed":False}
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",type=Path,required=True)
    p.add_argument("--horizon",type=float,default=2.)
    p.add_argument("--pulse",type=float,default=1.5)
    p.add_argument("--angle",type=float,default=30.)
    args=p.parse_args()
    print(json.dumps(run(args.files,args.horizon,args.pulse,args.angle),
                     indent=2,ensure_ascii=False,allow_nan=False))

