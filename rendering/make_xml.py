import xml.etree.ElementTree as et
import xml.dom.minidom
import subprocess
import os
from glob import glob
import re
output_type = "./"

# data_list = "../example/test/20141030082626_flip0_expo-1.jpg"

dir_path = "/media/shin/2nd_m.2/learningHDR_data/synthetic_data/train/"
hdr_path = str(dir_path+"/*.exr")
data_list = glob(hdr_path)
data_list.sort()

for data in data_list:

    m = re.split(dir_path, data)
    m = re.split(".exr", m[1])[0]

    scene = et.Element("scene", version = "2.0.0")

    integrator = et.SubElement(scene, "integrator", name="integrator", type = "direct")
    et.SubElement(integrator, "boolean", name = "hide_emitters", value = "false")
    # et.SubElement(integrator, "boolean",name = "strictNormals", value = "false")

    sensor = et.SubElement(scene,"sensor", type = "perspective")
    et.SubElement(sensor,"float", name = "far_clip", value = "100.0")
    et.SubElement(sensor, "float", name = "focus_distance", value = "23.4563")
    et.SubElement(sensor, "float", name = "fov", value = "20.3117")
    et.SubElement(sensor,"string", name = "fov_axis", value = "x")
    et.SubElement(sensor, "float", name = "near_clip", value = "0.1")

    sensor_transform = et.SubElement(sensor, "transform", name = "to_world")
    et.SubElement(sensor_transform, "lookat", origin="11.4325, 21.8848, 1.72842", target="11.4672, 20.9288, 2.01965", up = "-0.00128896, -0.291446, -0.956587")
    sampler = et.SubElement(sensor, "sampler", type = "ldsampler")
    et.SubElement(sampler,"integer", name = "sample_count", value = "256")

    film = et.SubElement(sensor, "film", type = "hdrfilm")
    et.SubElement(film, "integer", name = "width", value = "64")
    et.SubElement(film, "integer", name = "height", value = "64")
    et.SubElement(film, "string", name = "pixel_format", value = "rgb")
    et.SubElement(film, "rfilter", type = "gaussian")
    # film = et.SubElement(sensor, "film", type = "ldrfilm")
    # et.SubElement(film, "boolean", name = "banner", value = "true")
    # et.SubElement(film, "float", name = "exposure", value = "0.0")
    # et.SubElement(film, "string", name = "fileFormat", value = "png")
    # et.SubElement(film, "boolean", name = "fitHorizontal", value = "true")
    # et.SubElement(film, "float", name = "gamma", value = "-1")
    # et.SubElement(film, "integer", name = "height", value = "64")
    # et.SubElement(film, "boolean", name = "high_qualityEdges", value = "false")
    # et.SubElement(film, "string", name = "label[-10,-10]", value = "")
    # et.SubElement(film, "float", name = "pixel_aspectX", value = "1")
    # et.SubElement(film, "float", name = "pixel_aspectY", value = "1")
    # et.SubElement(film, "string", name = "pixel_format", value = "rgb")
    # et.SubElement(film, "float", name = "shiftX", value = "0")
    # et.SubElement(film, "float", name = "shiftY", value = "0")
    # et.SubElement(film, "string", name = "tonemap_method", value = "gamma")
    # et.SubElement(film, "integer", name = "width", value = "64")
    # rfilter = et.SubElement(film, "rfilter", type = "gaussian")
    # et.SubElement(rfilter, "float", name = "stddev", value = "0.5")

    obj_bsdf = et.SubElement(scene, "bsdf", id = "diffuse.003-bl_mat-bsdf", type = "diffuse")
    et.SubElement(obj_bsdf, "rgb", name = "reflectance", value = "0.288298, 0.288298, 0.288298")

    obj = et.SubElement(scene, "shape", id = "bunny.003_bunny.003_0000_m000_0.000000", type = "serialized")
    obj_transform = et.SubElement(obj, "transform", name = "to_world")
    et.SubElement(obj_transform, "matrix", value = "1.962585 0.000000 0.385044 12.443063 0.000000 2.000000 -0.000000 -5.115613 -0.385044 0.000000 1.962585 10.234327 0.000000 0.000000 0.000000 1.000000" )
    et.SubElement(obj, "ref", name = "bsdf", id = "diffuse.003-bl_mat-bsdf")
    et.SubElement(obj,"string", name = "filename", value = "./bunny_003_0000_m000_0_000000.serialized")

    emitter = et.SubElement(scene,"emitter", type = "envmap")

    emitter_transform = et.SubElement(emitter, "transform", name = "to_world")
    et.SubElement(emitter_transform, "matrix", value = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")
    et.SubElement(emitter_transform, "rotate", y="1", angle = "0")

    # et.SubElement(emitter, "rgb", name = "irradiance", value = "1.000000 1.000000 1.000000")
    et.SubElement(emitter, "float", name = "scale", value = "1")
    et.SubElement(emitter,"string", name = "filename", value = data)

    plane_bsdf = et.SubElement(scene, "bsdf", id = "diffuse_ground.003-bl_mat-bsdf", type = "diffuse")
    et.SubElement(plane_bsdf, "rgb", name = "reflectance", value = "0.114435 0.097587 0.082283")

    plane = et.SubElement(scene, "shape", id = "Plane.003_Plane.003_0000_m000_0.000000", type = "serialized")
    plane_transform = et.SubElement(plane, "transform", name = "to_world")
    et.SubElement(plane_transform, "matrix", value = "7.000000 0.000000 0.000000 12.565207 0.000000 0.000000 11.000000 -6.645764 0.000000 -14.150000 0.000000 3.963688 0.000000 0.000000 0.000000 1.000000")

    et.SubElement(plane, "ref", name = "bsdf", id = "diffuse_ground.003-bl_mat-bsdf")
    et.SubElement(plane, "string", name = "filename", value = "./Plane_003_0000_m000_0_000000.serialized")

    rough_string = et.tostring(scene,"utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    reparsed_pretty = reparsed.toprettyxml(indent=""*4)

    with open("aa.xml","w") as render_xml:
        render_xml.write(reparsed_pretty)

    output_type = output_type+"render"

    # +"%d"%(iteration)

    rough_string = et.tostring(scene,"utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)

    reparsed_pretty = reparsed.toprettyxml(indent=""*4)

    print(reparsed_pretty)

    with open("aa.xml","w") as cube_xml:
        cube_xml.write(reparsed_pretty)

    subprocess.check_output(["python", "render_scene.py", m])

# class render_scene(metaclass=dp.singleton):
    
#     def __init__(self):
#         # Absolute or relative path to the XML file
#         pass
        
#     def __call__(self, *args, **kwds):
#         pass
