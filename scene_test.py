import xml.etree.ElementTree as et
import xml.dom.minidom
import subprocess


def make_scene(envmap, iteration, flag):
    if flag==True:
        output_type = "images/label/"
    else:
        output_type = "images/pred/"

    
    data_list = envmap.decode('utf-8')
    
    scene = et.Element("scene", version = "0.6.0")

    integrator = et.SubElement(scene, "integrator", name="integrator", type = "direct")
    et.SubElement(integrator, "integer", name ="bsdfSamples", value = "1")
    et.SubElement(integrator, "integer", name = "emitterSamples", value = "1")
    et.SubElement(integrator, "boolean", name = "hideEmitters", value = "false")
    et.SubElement(integrator, "boolean",name = "strictNormals", value = "false")


    sensor = et.SubElement(scene,"sensor", type = "perspective")
    et.SubElement(sensor,"float", name = "farClip", value = "100.0")
    et.SubElement(sensor, "float", name = "focusDistance", value = "23.4563")
    et.SubElement(sensor, "float", name = "fov", value = "20.3117")
    et.SubElement(sensor,"string", name = "fovAxis", value = "x")
    et.SubElement(sensor, "float", name = "nearClip", value = "0.1")

    sensor_transform = et.SubElement(sensor, "transform", name = "toWorld")
    et.SubElement(sensor_transform, "lookat", origin="11.4325, 21.8848, 1.72842", target="11.4672, 20.9288, 2.01965", up = "-0.00128896, -0.291446, -0.956587")
    sampler = et.SubElement(sensor, "sampler", type = "ldsampler")
    et.SubElement(sampler,"integer", name = "sampleCount", value = "128")

    film = et.SubElement(sensor, "film", type = "ldrfilm")
    et.SubElement(film, "boolean", name = "banner", value = "true")
    et.SubElement(film, "float", name = "exposure", value = "0.0")
    et.SubElement(film, "string", name = "fileFormat", value = "png")
    et.SubElement(film, "boolean", name = "fitHorizontal", value = "true")
    et.SubElement(film, "float", name = "gamma", value = "-1")
    et.SubElement(film, "integer", name = "height", value = "64")
    et.SubElement(film, "boolean", name = "highQualityEdges", value = "false")
    et.SubElement(film, "string", name = "label[-10,-10]", value = "")
    et.SubElement(film, "float", name = "pixelAspectX", value = "1")
    et.SubElement(film, "float", name = "pixelAspectY", value = "1")
    et.SubElement(film, "string", name = "pixelFormat", value = "rgb")
    et.SubElement(film, "float", name = "shiftX", value = "0")
    et.SubElement(film, "float", name = "shiftY", value = "0")
    et.SubElement(film, "string", name = "tonemapMethod", value = "gamma")
    et.SubElement(film, "integer", name = "width", value = "64")
    rfilter = et.SubElement(film, "rfilter", type = "gaussian")
    et.SubElement(rfilter, "float", name = "stddev", value = "0.5")

    obj_bsdf = et.SubElement(scene, "bsdf", id = "diffuse.003-bl_mat-bsdf", type = "diffuse")
    et.SubElement(obj_bsdf, "rgb", name = "reflectance", value = "0.288298, 0.288298, 0.288298")

    obj = et.SubElement(scene, "shape", id = "bunny.003_bunny.003_0000_m000_0.000000", type = "serialized")
    obj_transform = et.SubElement(obj, "transform", name = "toWorld")
    et.SubElement(obj_transform, "matrix", value = "1.962585 0.000000 0.385044 12.443063 0.000000 2.000000 -0.000000 -5.115613 -0.385044 0.000000 1.962585 10.234327 0.000000 0.000000 0.000000 1.000000" )
    et.SubElement(obj, "ref", name = "bsdf", id = "diffuse.003-bl_mat-bsdf")
    et.SubElement(obj,"string", name = "filename", value = "/scene/00001/bunny_003_0000_m000_0_000000.serialized")


    emitter = et.SubElement(scene,"emitter", type = "envmap")

    emitter_transform = et.SubElement(emitter, "transform", name = "toWorld")
    et.SubElement(emitter_transform, "matrix", value = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")
    et.SubElement(emitter_transform, "rotate", y="1", angle = "0")

    et.SubElement(emitter, "rgb", name = "irradiance", value = "1.000000 1.000000 1.000000")
    et.SubElement(emitter, "float", name = "scale", value = "1")
    et.SubElement(emitter,"string", name = "filename", value = data_list)


    plane_bsdf = et.SubElement(scene, "bsdf", id = "diffuse_ground.003-bl_mat-bsdf", type = "diffuse")
    et.SubElement(plane_bsdf, "rgb", name = "reflectance", value = "0.114435 0.097587 0.082283")

    plane = et.SubElement(scene, "shape", id = "Plane.003_Plane.003_0000_m000_0.000000", type = "serialized")
    plane_transform = et.SubElement(plane, "transform", name = "toWorld")
    et.SubElement(plane_transform, "matrix", value = "7.000000 0.000000 0.000000 12.565207 0.000000 0.000000 11.000000 -6.645764 0.000000 -14.150000 0.000000 3.963688 0.000000 0.000000 0.000000 1.000000")

    et.SubElement(plane, "ref", name = "bsdf", id = "diffuse_ground.003-bl_mat-bsdf")
    et.SubElement(plane, "string", name = "filename", value = "/scene/00001/Plane_003_0000_m000_0_000000.serialized")

    rough_string = et.tostring(scene,"utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    reparsed_pretty = reparsed.toprettyxml(indent=""*4)

    with open("render.xml","w") as render_xml:
        render_xml.write(reparsed_pretty)
    
    output_type = output_type+"render"+"%d"%(iteration)

    subprocess.check_output(["mitsuba", "-o",output_type,"render.xml"])