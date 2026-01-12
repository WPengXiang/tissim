from fealpy.backend import bm
from fealpy.mesh import TriangleMesh

import matplotlib.pyplot as plt

def generate_mesh(box, halos, h, singular_points, hs, is_bspline=False):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("box_with_halo_tri")

    # 设置全局网格尺寸选项
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(hs) / 10)  # 允许小尺寸
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)  # 禁用曲率自适应

    # 创建大矩形
    box_gmsh = gmsh.model.occ.addRectangle(box[0], box[2], 0, box[1] - box[0], box[3] - box[2])

    # 创建 halo 边界：用 BSpline 拟合所有点为单条闭合曲线
    point_tags = []
    for p in halos:
        point_tags.append(gmsh.model.occ.addPoint(p[0], p[1], 0, h))  # 设置 halo 点默认尺寸为 h
    if is_bspline:
        point_tags_closed = point_tags + [point_tags[0]]  # 闭合
        spline_tag = gmsh.model.occ.addBSpline(point_tags_closed, degree=3)  # 立方 B-样条
        halo_curve_loop = gmsh.model.occ.addCurveLoop([spline_tag])
        halo_surface = gmsh.model.occ.addPlaneSurface([halo_curve_loop])
        gmsh.model.occ.remove([(0, tag) for tag in point_tags_closed], recursive=False)
    else:
        line_tags = []
        for i, p in enumerate(point_tags):
            # 首尾相连：最后一个点连接到第一个点
            p1 = point_tags[i]
            p2 = point_tags[(i + 1) % len(point_tags)]  # 模运算确保闭合
            line = gmsh.model.occ.addLine(p1, p2)
            line_tags.append(line)
        halo_curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
        halo_surface = gmsh.model.occ.addPlaneSurface([halo_curve_loop])

    # 创建奇异点
    singular_point_tags = []
    for p in singular_points:
        singular_point_tags.append(gmsh.model.occ.addPoint(p[0], p[1], 0))

    # 几何差集
    domain_tag, _ = gmsh.model.occ.cut([(2, box_gmsh)], [(2, halo_surface)])

    # 嵌入奇异点
    gmsh.model.occ.fragment(domain_tag, [(0, tag) for tag in singular_point_tags])
    gmsh.model.occ.synchronize()

    # 为奇异点设置局部网格加密
    fields = []  # 收集所有 Threshold Field
    for i, sp in enumerate(singular_point_tags):
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [sp])

        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", hs[i])
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 0.2)  # 缩小加密范围

        fields.append(f_thresh)

    # 创建 Min Field 合并所有 Threshold Field
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)

    # 设置背景网格
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # 生成网格
    gmsh.model.mesh.generate(2)

    node_tags, nodes, _ = gmsh.model.mesh.getNodes()
    node = bm.array(nodes, dtype=bm.float64).reshape(-1, 3)[:, :2]

    cell_type, cell_tags, cells = gmsh.model.mesh.getElements(2, -1)
    cell = bm.array(cells, dtype=bm.int64).reshape(-1, 3) - 1

    gmsh.finalize()

    tri_mesh = TriangleMesh(node, cell)
    return tri_mesh


if __name__ == '__main__':
    box = [-0.5, 1.5, -0.3, 0.3]
    h = 0.1
    halos = [
        [1.0000,    0.00120],
        [0.9500,     0.01027],
        [0.9000,     0.01867],
        [0.8000,     0.03320],
        [0.7000,     0.04480],
        [0.6000,     0.05320],
        [0.5000,     0.05827],
        [0.4000,     0.06000],
        [0.3000,     0.05827],
        [0.2000,     0.05293],
        [0.1500,     0.04867],
        [0.1000,     0.04240],
        [0.0750,     0.03813],
        [0.0500,     0.03267],
        [0.0250,     0.02453],
        [0.0125,     0.01813],
        [0.0000,     0.00000],
        [0.0125,     -0.01813],
        [0.0250,     -0.02453],
        [0.0500,     -0.03267],
        [0.0750,     -0.03813],
        [0.1000,     -0.04240],
        [0.1500,     -0.04867],
        [0.2000,     -0.05293],
        [0.3000,     -0.05827],
        [0.4000,     -0.06000],
        [0.5000,     -0.05827],
        [0.6000,     -0.05320],
        [0.7000,     -0.04480],
        [0.8000,     -0.03320],
        [0.9000,     -0.01867],
        [0.9500,     -0.01027],
        [1.0000,     -0.00120]]
    singular_points = [[0, 0], [1, 0]]
    hs = [0.01, 0.005]  # 奇异点处网格尺寸

    mesh = generate_mesh(box, halos, h, singular_points, hs, is_bspline=False)

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    plt.show()



