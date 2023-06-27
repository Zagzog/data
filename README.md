# Network data and program

main_program.py
The simulation program of our model, where h=1

main_program_support.py
The auxiliary programs of the main_program.py include network generation, initial opinion generation, and selection of extreme individuals

theoretical_simulation_analysis.py
Theoretical calculation program, including:
1.theoretical opinion distribution (theory_of_p and theory_frequency)
2.assortative coefficients(calculate_TCI)
3.opinion distribution of simulation results(simulation_frequency)
4.the ratio of edge overlapand (edge_overlap)
5.calculate the spread range results of different h(change_h_data_clean)
6.calculate the opinion results of different h(simulation_frequency_h)
7.networks structure required for visualization(edge_select)

control_and_experimental_scheme.py
The simulation program of control scheme and experimental scheme

information-opinion-add.py
The simulation program of our model with different h(h=0.1,1,10,100)

figure_plot.py
Program for all figures
