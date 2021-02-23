import os
import subprocess
import numpy as np
import itertools
import collections
import shutil
import csv
import ruamel.yaml as yaml
from typing import List

from bgem.gmsh import gmsh
from bgem.gmsh import gmsh_io
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import heal_mesh

from mlmc.level_simulation import LevelSimulation
from mlmc.sim.simulation import Simulation
from mlmc.sim.simulation import QuantitySpec

import matplotlib.pyplot as plt


def force_mkdir(path, force=False):
    """
    Make directory 'path' with all parents,
    remove the leaf dir recursively if it already exists.
    :param path: path to directory
    :param force: if dir already exists then remove it and create new one
    :return: None
    """
    if force:
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(path, mode=0o775, exist_ok=True)


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


class endorse_2Dtest(Simulation):

    zero_temperature_offset = 273.15

    def __init__(self, config, clean):
        super(endorse_2Dtest, self).__init__()

        # TODO: how should I know, that these variables must be set here ?
        self.need_workspace = True
        self.work_dir = config["work_dir"]
        self.clean = clean

        self._config = config

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float])-> LevelSimulation:
        """
        Overrides Simulation.level_instance
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.)
        :param fine_level_params: in this version, it is just fine simulation step
        :param coarse_level_params: in this version, it is just coarse simulation step
        :return: mlmc.LevelSimulation object, this object is serialized in SamplingPoolPbs and deserialized in PbsJob,
         so it allows pass simulation data from main process to PBS process
        """

        config = self._config.copy()

        # Set fine simulation common files directory
        # Files in the directory are used by each simulation at that level
        common_files_dir = os.path.join(self.work_dir, "common_files")
        force_mkdir(common_files_dir, force=self.clean)
        config["common_files_dir"] = common_files_dir

        # copy common files
        for f in config["copy_files"]:
            shutil.copyfile(os.path.join(config["script_dir"], f), os.path.join(common_files_dir, f))

        return LevelSimulation(config_dict=config,
                               # task_size=len(fine_mesh_data['points']),
                               task_size=config["task_size"],
                               calculate=endorse_2Dtest.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True  # If True, a sample directory is created
                               )

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Overrides Simulation.result_format
        :return:
        """
        # create simple instance of QuantitySpec for each quantity we want to collect
        # the time vector of the data must be specified here!

        # TODO: define times according to output times of Flow123d
        # TODO: how should be units defined (and other members)?
        step = 1
        end_time = 31
        times = list(range(0, end_time, step))
        spec = []
        spec.append(QuantitySpec(name="avg_temp_02", unit="C", shape=(1, 1), times=times, locations=['.well']))
        spec.append(QuantitySpec(name="power_02", unit="J", shape=(1, 1), times=times, locations=['.well']))
        spec.append(QuantitySpec(name="avg_temp_03", unit="C", shape=(1, 1), times=times, locations=['.well']))
        spec.append(QuantitySpec(name="power_03", unit="J", shape=(1, 1), times=times, locations=['.well']))
        spec.append(QuantitySpec(name="avg_temp_04", unit="C", shape=(1, 1), times=times, locations=['.well']))
        spec.append(QuantitySpec(name="power_04", unit="J", shape=(1, 1), times=times, locations=['.well']))
        spec.append(QuantitySpec(name="n_fracture_elements", unit="-", shape=(1, 1), times=[0], locations=['-']))
        spec.append(QuantitySpec(name="n_contact_elements", unit="-", shape=(1, 1), times=[0], locations=['-']))
        return spec

    @staticmethod
    def calculate(config_dict, seed):
        """
        Overrides Simulation.calculate. The program is currently in <work_dir>/<sample_id_dir> directory.
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config_dict: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        # does all the data preparation, passing
        # running simulation
        # extracting results

        print("=========================== RUNNING CALCULATION ===========================")

        # Set random seed, seed is calculated from sample id, so it is not user defined
        np.random.seed(seed)

        # collect only
        if config_dict["collect_only"]:
            return endorse_2Dtest.collect_results(config_dict)

        mesh_repo = config_dict.get('mesh_repository', None)

        print("Creating mesh...")
        if mesh_repo:
            comp_mesh = endorse_2Dtest.sample_mesh_repository(mesh_repo)
        else:
            # comp_mesh = endorse_2Dtest.prepare_mesh(config_dict, cut_tunnel=False)
            comp_mesh = endorse_2Dtest.prepare_mesh(config_dict, cut_tunnel=True)

            mesh_bn = os.path.basename(comp_mesh)
            config_dict["hm_params"]["mesh"] = mesh_bn

        # endorse_2Dtest.read_physical_names(config_dict, comp_mesh)
        print("Creating mesh...finished")

        if config_dict["mesh_only"]:
            return endorse_2Dtest.empty_result()

        # endorse_2Dtest.prepare_hm_input(config_dict)
        print("Running Flow123d - HM...")
        hm_succeed = endorse_2Dtest.call_flow(config_dict, 'hm_params', result_files=["flow_observe.yaml"])
        if not hm_succeed:
            raise Exception("HM model failed.")
        print("Running Flow123d - HM...finished")

        endorse_2Dtest.observe_time_plot(config_dict)

        print("Finished computation")

        return endorse_2Dtest.collect_results(config_dict)

    @staticmethod
    def check_data(data, minimum, maximum):
        n_times = len(endorse_2Dtest.result_format()[0].times)
        if len(data) != n_times:
            raise Exception("Data not corresponding with time axis.")

        if np.isnan(np.sum(data)):
            raise Exception("NaN present in extracted data.")

        min = np.amin(data)
        if min < minimum:
            raise Exception("Data out of given range [min].")
        max = np.amax(data)
        if max > maximum:
            raise Exception("Data out of given range [max].")

    @staticmethod
    def collect_results(config_dict):
        filenames = ["energy_balance.yaml",
                     "water_balance.yaml",
                     "Heat_AdvectionDiffusion_region_stat.yaml"]

        print("Extracting results...")
        # read regions of interest
        # bc_regions = ['.fr_left_well', '.left_well', '.fr_right_well', '.right_well']
        # out_regions = bc_regions[2:]
        with open("regions.yaml", 'r') as f:
            regions_dict = yaml.load(f, yaml.CSafeLoader)
            bc_regions = [*regions_dict["left_well_fracture_regions"], *regions_dict["right_well_fracture_regions"]]
            out_regions = regions_dict["right_well_fracture_regions"]

        n_times = 0
        result = []
        for variant in config_dict["variants"]:
            print("collecting TH - variant '{}'...".format(variant))
            config_dict[variant]["output_dir"] = "output_" + config_dict[variant]["in_file"]

            result_files = list()
            result_files.extend([os.path.join(config_dict[variant]["output_dir"], f) for f in filenames])
            files_present = all([os.path.isfile(f) for f in result_files])
            if not files_present:
                raise Exception("Not all result files present.")
            avg_temp, power = endorse_2Dtest.extract_th_results(config_dict[variant]["output_dir"],
                                                                  out_regions, bc_regions)
            n_times=len(power)
            endorse_2Dtest.check_data(avg_temp, config_dict["extract"]["temp_min"],
                                        config_dict["extract"]["temp_max"])
            endorse_2Dtest.check_data(power, config_dict["extract"]["power_min"],
                                        config_dict["extract"]["power_max"])

            result.extend(avg_temp)
            result.extend(power)

        fr_file = "fr_param_stats.yaml"
        if os.path.isfile(fr_file):
            with open(fr_file, 'r') as f:
                fr_param_dict = yaml.load(f, yaml.CSafeLoader)
                result.extend([fr_param_dict["n_fracture_elements"]]*n_times)
                result.extend([fr_param_dict["n_contact_elements"]]*n_times)
        else:
            raise Exception("Fracture stats file '{}' not present.".format(fr_file))

        print("Extracting results...finished")

        return [result, result]

    @staticmethod
    def empty_result():
        return [[np.random.normal()], [np.random.normal()]]

    @staticmethod
    def check_conv_reasons(log_fname):
        with open(log_fname, "r") as f:
            for line in f:
                tokens = line.split(" ")
                try:
                    i = tokens.index('convergence')
                    if tokens[i + 1] == 'reason':
                        value = tokens[i + 2].rstrip(",")
                        conv_reason = int(value)
                        if conv_reason < 0:
                            print("Failed to converge: ", conv_reason)
                            return False
                except ValueError:
                    continue
        return True

    @staticmethod
    def call_flow(config_dict, param_key, result_files):
        """
        Redirect sstdout and sterr, return true on succesfull run.
        :param result_files: Files to be computed - skip computation if already exist.
        :param param_key: config dict parameters key
        :param config_dict:
        :return:
        """

        params = config_dict[param_key]
        fname = params["in_file"]
        arguments = config_dict["_aux_flow_path"].copy()
        output_dir = "output_" + fname
        config_dict[param_key]["output_dir"] = output_dir
        if all([os.path.isfile(os.path.join(output_dir, f)) for f in result_files]):
            status = True
        else:
            substitute_placeholders(os.path.join(config_dict["common_files_dir"], fname + '_tmpl.yaml'),
                                    fname + '.yaml',
                                    params)
            arguments.extend(['--no_profiler', '--output_dir', output_dir, fname + ".yaml"])
            print("Running: ", " ".join(arguments))
            with open(fname + "_stdout", "w") as stdout:
                with open(fname + "_stderr", "w") as stderr:
                    completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0
        conv_check = endorse_2Dtest.check_conv_reasons(os.path.join(output_dir, "flow123.0.log"))
        print("converged: ", conv_check)
        return status and conv_check










    @staticmethod
    def sample_mesh_repository(mesh_repository):
        """
        Select random mesh from the given mesh repository.
        """
        mesh_file = np.random.choice(os.listdir(mesh_repository))
        comp_mesh = "random_fractures_healed.msh"
        shutil.copyfile(os.path.join(mesh_repository, mesh_file), comp_mesh)
        # heal_ref_report = {'flow_stats': {'bad_el_tol': 0.01, 'bad_elements': [], 'bins': [], 'hist': []},
        #                    'gamma_stats': {'bad_el_tol': 0.01, 'bad_elements': [], 'bins': [], 'hist': []}}
        # with open("random_fractures_heal_stats.yaml", "w") as f:
        #     yaml.dump(heal_ref_report, f)
        return comp_mesh

    # @staticmethod
    # def read_physical_names(config_dict, mesh_file):
    #     """
    #     Read physical names to set fracture B.C. regions in Flow123d yaml later.
    #     :param config_dict:
    #     :param mesh_file:
    #     :return:
    #     """
    #     mesh_bn = os.path.basename(mesh_file)
    #     reader = gmsh_io.GmshIO()
    #     reader.filename = mesh_file
    #     regions = [*reader.read_physical_names()]
    #
    #     # fracture regions
    #     reg_fr = [reg for reg in regions if reg.startswith("fr")]
    #
    #     # split boundary fracture regions on left and right well
    #     reg_fr_left_well = [reg for reg in regions if ".fr" in reg and "left_well" in reg]
    #     reg_fr_right_well = [reg for reg in regions if ".fr" in reg and "right_well" in reg]
    #     for model in ["hm_params", *config_dict["variants"]]:
    #         model_dict = config_dict[model]
    #         model_dict["mesh"] = mesh_bn
    #         model_dict["fracture_regions"] = reg_fr
    #         model_dict["left_well_fracture_regions"] = reg_fr_left_well
    #         model_dict["right_well_fracture_regions"] = reg_fr_right_well
    #
    #     # write grouped physical names into file for later usage (collection when having no mesh available anymore)
    #     regions_dict = dict()
    #     regions_dict["fracture_regions"] = reg_fr
    #     regions_dict["left_well_fracture_regions"] = reg_fr_left_well
    #     regions_dict["right_well_fracture_regions"] = reg_fr_right_well
    #     with open('regions.yaml', 'w') as outfile:
    #         yaml.dump(regions_dict, outfile, default_flow_style=False, Dumper=yaml.CDumper)

    @staticmethod
    def prepare_mesh(config_dict, cut_tunnel):
        mesh_name = config_dict["mesh_name"]
        if cut_tunnel:
            mesh_name = mesh_name + "_cut"
        mesh_file = mesh_name + ".msh"
        if not os.path.isfile(mesh_file):
            endorse_2Dtest.make_mesh(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)

        mesh_healed = mesh_name + "_healed.msh"
        if not os.path.isfile(mesh_healed):
            hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
            hm.heal_mesh(gamma_tol=0.01)
            hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
            hm.write()
            assert hm.healed_mesh_name == mesh_healed
        return mesh_healed

    @staticmethod
    def make_mesh(config_dict, mesh_name, mesh_file, cut_tunnel):
        geom = config_dict["geometry"]
        tunnel_mesh_step = geom['tunnel_mesh_step']
        dimensions = geom["box_dimensions"]
        tunnel_dims = np.array(geom["tunnel_dimensions"])/2
        tunnel_center = geom["tunnel_center"]

        print("load gmsh api")
        factory = gmsh.GeometryOCC(mesh_name, verbose=True)
        gmsh_logger = factory.get_logger()
        gmsh_logger.start()
        gopt = gmsh_options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001
        # gopt.MatchMeshTolerance = 1e-1
        gopt.OCCFixSmallEdges = True
        gopt.OCCFixSmallFaces = True

        # Main box
        box = factory.rectangle(dimensions).set_region("box")
        side = factory.line([-dimensions[0]/2, 0, 0], [dimensions[0]/2, 0, 0])
        sides = dict(
            bottom=side.copy().translate([0, -dimensions[1] / 2, 0]),
            top   =side.copy().translate([0, +dimensions[1] / 2, 0]),
            left  =side.copy().translate([0, +dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2),
            right =side.copy().translate([0, -dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2)
        )

        # dir_factor = 0.01
        # d = dir_factor * np.array(dimensions)
        # # dir_side = factory.line([0, 0, 0], [dimensions[0] * dir_factor, 0, 0])
        # # dir_side = dir_side.translate([-dimensions[0] / 2, 0, 0])
        # dir_sides = dict(
        #     fix_bottom=factory.line([0, 0, 0], [d[0], 0, 0]).translate([-dimensions[0] / 2, -dimensions[1] / 2, 0]),
        #     bottom=factory.line([0, 0, 0], [dimensions[0] - d[0], 0, 0]).translate([-dimensions[0] / 2 + d[0], -dimensions[1] / 2, 0]),
        #     fix_left=factory.line([0, 0, 0], [0, d[1], 0]).translate([-dimensions[0] / 2, -dimensions[1] / 2, 0]),
        #     left = factory.line([0, 0, 0], [0, dimensions[1] - d[1], 0]).translate([-dimensions[0] / 2, -dimensions[1] / 2 + d[1], 0])
        #     # fix_bottom=dir_side.copy().translate([0, -dimensions[1] / 2, 0]),
        #     # fix_left  =dir_side.copy().translate([0, +dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2),
        # )
        # dir_sides_select = {key: value.copy() for key, value in dir_sides.items()}

        tunnel_disc = factory.disc(tunnel_center, *tunnel_dims)
        tunnel_select = tunnel_disc.copy()

        print("cutting and fragmenting...")
        # drilled box and its boundary
        box_drilled = box.cut(tunnel_disc)

        # # fractures, fragmented, fractures boundary
        # print("cut fractures by box without wells...")
        # fractures_shapes_group = fractures_shapes_group.intersect(box_drilled.copy())
        # print("fragment fractures...")
        # dlist = factory.fragment(box_drilled, tunnel_disc, *[v for v in dir_sides.values()])
        # box_fr = dlist[0]
        # tunnel_fr = dlist[1]
        box_fr, tunnel_fr = factory.fragment(box_drilled, tunnel_disc)

        print("marking boundary regions...")
        box_all = []

        b_box_fr = box_fr.get_boundary()
        for name, side_tool in sides.items():
            isec = b_box_fr.select_by_intersect(side_tool)
            box_all.append(isec.modify_regions("." + name))
        # for name, side_tool in dir_sides_select.items():
        #     isec = b_box_fr.select_by_intersect(side_tool)
        #     box_all.append(isec.modify_regions("." + name))

        if cut_tunnel:
            b_tunnel_select = tunnel_select.get_boundary()
            b_tunnel = b_box_fr.select_by_intersect(b_tunnel_select)
            b_tunnel.modify_regions(".tunnel").mesh_step(tunnel_mesh_step)
            box_all.extend([box_fr, b_tunnel])
        else:
            tunnel = tunnel_fr.select_by_intersect(tunnel_select)
            tunnel.set_region("tunnel").mesh_step(tunnel_mesh_step)
            box_all.extend([box_fr, tunnel])

        mesh_groups = [*box_all]

        print("meshing...")

        factory.keep_only(*mesh_groups)
        factory.remove_duplicate_entities()
        factory.write_brep()

        min_el_size = tunnel_mesh_step / 2
        max_el_size = np.max(dimensions) / 10

        mesh = gmsh_options.Mesh()
        # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
        # mesh.Algorithm = options.Algorithm2d.Delaunay
        # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
        # mesh.Algorithm3D = options.Algorithm3d.Frontal
        # mesh.Algorithm3D = options.Algorithm3d.Delaunay

        # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
        mesh.Algorithm3D = gmsh_options.Algorithm3d.HXT

        mesh.ToleranceInitialDelaunay = 0.01
        # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
        mesh.CharacteristicLengthFromPoints = True
        mesh.CharacteristicLengthFromCurvature = True
        mesh.CharacteristicLengthExtendFromBoundary = 2
        mesh.CharacteristicLengthMin = min_el_size
        mesh.CharacteristicLengthMax = max_el_size
        mesh.MinimumCirclePoints = 6
        mesh.MinimumCurvePoints = 2

        # factory.make_mesh(mesh_groups, dim=2)
        factory.make_mesh(mesh_groups)

        gmsh_log_msgs = gmsh_logger.get()
        gmsh_logger.stop()
        endorse_2Dtest.check_gmsh_log(gmsh_log_msgs)

        factory.write_mesh(format=gmsh.MeshFormat.msh2)
        os.rename(mesh_name + ".msh2", mesh_file)
        # factory.show()

    @staticmethod
    def check_gmsh_log(lines):
        """
        Search for "No elements in volume" message -> could not mesh the volume -> empty mesh.
        # PLC Error:  A segment and a facet intersect at point (-119.217,65.5762,-40.8908).
        #   Segment: [70,2070] #-1 (243)
        #   Facet:   [3147,9829,13819] #482
        # Info    : failed to recover constrained lines/triangles
        # Info    : function failed
        # Info    : function failed
        # Error   : HXT 3D mesh failed
        # Error   : No elements in volume 1
        # Info    : Done meshing 3D (Wall 0.257168s, CPU 0.256s)
        # Info    : 13958 nodes 34061 elements
        # Error   : ------------------------------
        # Error   : Mesh generation error summary
        # Error   :     0 warnings
        # Error   :     2 errors
        # Error   : Check the full log for details
        # Error   : ------------------------------
        """
        empty_volume_error = "No elements in volume"
        res = [line for line in lines if empty_volume_error in line]
        if len(res) != 0:
            raise Exception("GMSH error - No elements in volume")

    # @staticmethod
    # def prepare_hm_input(config_dict):
    #     """
    #     Prepare FieldFE input file for the TH simulation.
    #     :param config_dict: Parsed config.yaml. see key comments there.
    #     """
    #     bc_pressure_csv = 'bc_pressure_tunnel.csv'
    #     if os.path.exists(bc_pressure_csv):
    #         return
    #
    #     end_time = 17
    #     time_step = 1
    #     times = np.arange(0, end_time, time_step)
    #     n_steps = len(times)
    #     times = np.append(times, end_time)
    #
    #     start_val = 300
    #     end_val = 0
    #     val_step = (end_val-start_val)/n_steps
    #     values = np.arange(start_val, end_val, val_step)
    #     values = np.append(values, end_val)
    #
    #     header = "time bc_pressure_tunnel"
    #     fmt = "%g"
    #     list_rows = np.column_stack((times, values))
    #     np.savetxt(bc_pressure_csv, list_rows, fmt=fmt, delimiter=' ', header=header)
    #     # with open(bc_pressure_csv, 'w', newline='') as csv_file:
    #     #     writer = csv.writer(csv_file)
    #     #     writer.writerow(["time", "bc_pressure_tunnel"])
    #     #     for t, v in zip(times, values):
    #     #         writer.writerow([t, v])

        # PREPARE normal traction on tunnel boundary evolving in time






    @staticmethod
    def extract_time_series(yaml_stream, regions, extract):
        """

        :param yaml_stream:
        :param regions:
        :return: times list, list: for every region the array of value series
        """
        data = yaml.load(yaml_stream, yaml.CSafeLoader)['data']
        times = set()
        reg_series = {reg: [] for reg in regions}

        for time_data in data:
            region = time_data['region']
            if region in reg_series:
                times.add(time_data['time'])
                power_in_time = extract(time_data)
                reg_series[region].append(power_in_time)
        times = list(times)
        times.sort()
        series = [np.array(region_series, dtype=float) for region_series in reg_series.values()]
        return np.array(times), np.array(series)

    @staticmethod
    def extract_th_results(output_dir, out_regions, bc_regions):
        with open(os.path.join(output_dir, "energy_balance.yaml"), "r") as f:
            power_times, reg_powers = endorse_2Dtest.extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
            power_series = -sum(reg_powers)

        with open(os.path.join(output_dir, "Heat_AdvectionDiffusion_region_stat.yaml"), "r") as f:
            temp_times, reg_temps = endorse_2Dtest.extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
        with open(os.path.join(output_dir, "water_balance.yaml"), "r") as f:
            flux_times, reg_fluxes = endorse_2Dtest.extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
        sum_flux = sum(reg_fluxes)

        reg_temps = reg_temps - endorse_2Dtest.zero_temperature_offset

        avg_temp_flux = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux
        return avg_temp_flux, power_series

    @staticmethod
    def plot_exchanger_evolution(temp_times, avg_temp, power_times, power_series):
        year_sec = 60 * 60 * 24 * 365

        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        temp_color = 'red'
        ax1.set_xlabel('time [y]')
        ax1.set_ylabel('Temperature [C deg]', color=temp_color)
        ax1.plot(temp_times[1:] / year_sec, avg_temp[1:], color=temp_color)
        ax1.tick_params(axis='y', labelcolor=temp_color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        pow_color = 'blue'
        ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
        ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
        ax2.tick_params(axis='y', labelcolor=pow_color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    @staticmethod
    def observe_time_plot(config_dict):

        output_dir = config_dict["hm_params"]["output_dir"]

        with open(os.path.join(output_dir, "flow_observe.yaml"), "r") as f:
            loaded_yaml = yaml.load(f, yaml.CSafeLoader)
            points = loaded_yaml['points']
            point_names = [p["name"] for p in points]
            data = loaded_yaml['data']
            values = np.array([d["pressure_p0"] for d in data]).transpose()
            times = np.array([d["time"] for d in data]).transpose()

            v = values[0, 0:]

            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots()
            temp_color = ['red', 'green', 'violet', 'blue']
            ax1.set_xlabel('time [d]')
            ax1.set_ylabel('pressure [m]')
            for i in range(0, len(point_names)):
                ax1.plot(times, values[i, 0:], color=temp_color[i], label=point_names[i])

            ax1.tick_params(axis='y')
            ax1.legend()

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.show()
            plt.savefig("observe_pressure.pdf")
