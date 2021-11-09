'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

# -*- coding: utf-8 -*-

import os, tarfile, numpy, math, networkx, queue, random,traceback
from enum import IntEnum

class TimeDist(IntEnum):
    """
    Enumeration of the supported time distributions 
    """
    EXPONENTIAL_T = 0
    DETERMINISTIC_T = 1
    UNIFORM_T = 2
    NORMAL_T = 3
    ONOFF_T = 4
    PPBP_T = 5
    
    @staticmethod
    def getStrig(timeDist):
        if (timeDist == 0):
            return ("EXPONENTIAL_T")
        elif (timeDist == 1):
            return ("DETERMINISTIC_T")
        elif (timeDist == 2):
            return ("UNIFORM_T")
        elif (timeDist == 3):
            return ("NORMAL_T")
        elif (timeDist == 4):
            return ("ONOFF_T")
        elif (timeDist == 5):
            return ("PPBP_T")
        else:
            return ("UNKNOWN")

class SizeDist(IntEnum):
    """
    Enumeration of the supported size distributions 
    """
    DETERMINISTIC_S = 0
    UNIFORM_S = 1
    BINOMIAL_S = 2
    GENERIC_S = 3
    
    @staticmethod
    def getStrig(sizeDist):
        if (sizeDist == 0):
            return ("DETERMINISTIC_S")
        elif (sizeDist == 1):
            return ("UNIFORM_S")
        elif (sizeDist == 2):
            return ("BINOMIAL_S")
        elif (sizeDist ==3):
            return ("GENERIC_S")
        else:
            return ("UNKNOWN")

class Sample:
    """
    Class used to contain the results of a single iteration in the dataset
    reading process.
    
    ...
    
    Attributes
    ----------
    global_packets : double
        Overall number of packets transmitteds in network
    global_losses : double
        Overall number of packets lost in network
    global_delay : double
        Overall delay in network
    maxAvgLambda: double
        This variable is used in our simulator to define the overall traffic 
        intensity  of the network scenario
    performance_matrix : NxN matrix
        Matrix where each cell [i,j] contains aggregated and flow-level
        information about transmission parameters between source i and
        destination j.
    traffic_matrix : NxN matrix
        Matrix where each cell [i,j] contains aggregated and flow-level
        information about size and time distributions between source i and
        destination j.
    routing_matrix : NxN matrix
        Matrix where each cell [i,j] contains the path, if it exists, between
        source i and destination j.
    topology_object : 
        Network topology using networkx format.
    """
    
    global_packets = None
    global_losses = None
    global_delay = None
    maxAvgLambda = None
    
    performance_matrix = None
    traffic_matrix = None
    routing_matrix = None
    topology_object = None
    
    _results_line = None
    _flowresults_line = None
    _routing_file = None
    _graph_file = None
    
    def get_global_packets(self):
        """
        Return the number of packets transmitted in the network per time unit of this Sample instance.
        """
        
        return self.global_packets

    def get_global_losses(self):
        """
        Return the number of packets dropped in the network per time unit of this Sample instance.
        """
        
        return self.global_losses
    
    def get_global_delay(self):
        """
        Return the average per-packet delay over all the packets transmitted in the network in time units 
        of this sample instance.
        """
        
        return self.global_delay
    
    def get_maxAvgLambda(self):
        """
        Returns the maxAvgLamda used in the current iteration. This variable is used in our simulator to define 
        the overall traffic intensity of the network scenario.
        """
        
        return self.maxAvgLambda
        
    def get_performance_matrix(self):
        """
        Returns the performance_matrix of this Sample instance.
        """
        
        return self.performance_matrix
    
    def get_srcdst_performance(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Result matrix for the requested src-dst.

        """
        return self.performance_matrix[src, dst]
        
    def get_traffic_matrix(self):
        """
        Returns the traffic_matrix of this Sample instance.
        """
        
        return self.traffic_matrix
    
    def get_srcdst_traffic(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Traffic matrix for the requested src-dst.

        """
        
        return self.traffic_matrix[src, dst]
        
    def get_routing_matrix(self):
        """
        Returns the routing_matrix of this Sample instance.
        """
        
        return self.routing_matrix
    
    def get_srcdst_routing(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Routing matrix for the requested src-dst.

        """
        return self.routing_matrix[src, dst]
        
    def get_topology_object(self):
        """
        Returns the topology in networkx format of this Sample instance.
        """
        
        return self.topology_object
    
    def get_network_size(self):
        """
        Returns the number of nodes of the topology.
        """
        return self.topology_object.number_of_nodes()
    
    def get_node_properties(self, id):
        """
        

        Parameters
        ----------
        id : int
            Node identifier.

        Returns
        -------
        Dictionary with the parameters of the node
        None if node doesn't exist

        """
        res = None
        
        if id in self.topology_object.nodes:
            res = self.topology_object.nodes[id] 
        
        return res
    
    def get_link_properties(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary with the parameters of the link
        None if no link exist between src and dst

        """
        res = None
        
        if dst in self.topology_object[src]:
            res = self.topology_object[src][dst][0] 
        
        return res
    
    def get_srcdst_link_bandwidth(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Bandwidth in bits/time unit of the link between nodes src-dst or -1 if not connected

        """
        if dst in self.topology_object[src]:
            cap = float(self.topology_object[src][dst][0]['bandwidth'])
        else:
            cap = -1
            
        return cap
        
        
    def _set_data_set_file_name(self,file):
        """
        Sets the data set file from where the sample is extracted.
        """
        self.data_set_file = file
        
    def _set_performance_matrix(self, m):
        """
        Sets the performance_matrix of this Sample instance.
        """
        
        self.performance_matrix = m
        
    def _set_traffic_matrix(self, m):
        """
        Sets the traffic_matrix of this Sample instance.
        """
        
        self.traffic_matrix = m
        
    def _set_routing_matrix(self, m):
        """
        Sets the traffic_matrix of this Sample instance.
        """
        
        self.routing_matrix = m
        
    def _set_topology_object(self, G):
        """
        Sets the topology_object of this Sample instance.
        """
        
        self.topology_object = G
        
    def _set_global_packets(self, x):
        """
        Sets the global_packets of this Sample instance.
        """
        
        self.global_packets = x
        
    def _set_global_losses(self, x):
        """
        Sets the global_losses of this Sample instance.
        """
        
        self.global_losses = x
        
    def _set_global_delay(self, x):
        """
        Sets the global_delay of this Sample instance.
        """
        
        self.global_delay = x
        
    def _get_data_set_file_name(self):
        """
        Gets the data set file from where the sample is extracted.
        """
        return self.data_set_file
    
    def _get_path_for_srcdst(self, src, dst):
        """
        Returns the path between node src and node dst.
        """
        
        return self.routing_matrix[src, dst]
    
    def _get_timedis_for_srcdst (self, src, dst):
        """
        Returns the time distribution of traffic between node src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['TimeDist']
    
    def _get_eqlambda_for_srcdst (self, src, dst):
        """
        Returns the equivalent lambda for the traffic between node src and node
        dst.
        """
        
        return self.traffic_matrix[src, dst]['EqLambda']
    
    def _get_timedistparams_for_srcdst (self, src, dst):
        """
        Returns the time distribution parameters for the traffic between node
        src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['TimeDistParams']
    
    def _get_sizedist_for_srcdst (self, src, dst):
        """
        Returns the size distribution of traffic between node src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['SizeDist']
    
    def _get_avgpktsize_for_srcdst_flow (self, src, dst):
        """
        Returns the average packet size for the traffic between node src and
        node dst.
        """
        
        return self.traffic_matrix[src, dst]['AvgPktSize']
    
    def _get_sizedistparams_for_srcdst (self, src, dst):
        """
        Returns the time distribution of traffic between node src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['SizeDistParams']
    
    def _get_resultdict_for_srcdst (self, src, dst):
        """
        Returns the dictionary with all the information for the communication
        between node src and node dst regarding communication parameters.
        """
        
        return self.performance_matrix[src, dst]
    
    def _get_trafficdict_for_srcdst (self, src, dst):
        """
        Returns the dictionary with all the information for the communication
        between node src and node dst regarding size and time distribution
        parameters.
        """
        
        return self.traffic_matrix[src, dst]

class DatanetAPI:
    """
    Class containing all the functionalities to read the dataset line by line
    by means of an iteratos, and generate a Sample instance with the
    information gathered.
    """
    
    def __init__ (self, data_folder, shuffle=False):
        """
        Initialization of the PasringTool instance

        Parameters
        ----------
        data_folder : str
            Folder where the dataset is stored.
        dict_queue : Queue
            Auxiliar data structures used to conveniently move information
            between the file where they are read, and the matrix where they
            are located.
        shuffle: boolean
            Specify if all files should be shuffled. By default false

        Returns
        -------
        None.

        """
        
        self.data_folder = data_folder
        self.dict_queue = queue.Queue()
        self.shuffle = shuffle


    def _readRoutingFile(self, routing_fd, netSize):
        """
        Pending to compare against getSrcPortDst

        Parameters
        ----------
        routing_file : str
            File where the routing information is located.
        netSize : int
            Number of nodes in the network.

        Returns
        -------
        R : netSize x netSize matrix
            Matrix where each  [i,j] states what port node i should use to
            reach node j.

        """
        
        R = numpy.zeros((netSize, netSize)) - 1
        src = 0
        for line in routing_fd:
            line = line.decode()
            camps = line.split(',')
            dst = 0
            for port in camps[:-1]:
                R[src][dst] = port
                dst += 1
            src += 1
        return (R)

    def _getRoutingSrcPortDst(self, G):
        """
        Return a dictionary of dictionaries with the format:
        node_port_dst[node][port] = next_node

        Parameters
        ----------
        G : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        node_port_dst = {}
        for node in G:
            port_dst = {}
            node_port_dst[node] = port_dst
            for destination in G[node].keys():
                port = G[node][destination][0]['port']
                node_port_dst[node][port] = destination
        return(node_port_dst)

    def _create_routing_matrix(self, G,routing_file):
        """

        Parameters
        ----------
        G : graph
            Graph representing the network.
        routing_file : str
            File where the information about routing is located.

        Returns
        -------
        MatrixPath : NxN Matrix
            Matrix where each cell [i,j] contains the path to go from node
            i to node j.

        """
        
        netSize = G.number_of_nodes()
        node_port_dst = self._getRoutingSrcPortDst(G)
        R = self._readRoutingFile(routing_file, netSize)
        MatrixPath = numpy.empty((netSize, netSize), dtype=object)
        for src in range (0,netSize):
            for dst in range (0,netSize):
                node = src
                path = [node]
                while (R[node][dst] != -1):
                    out_port = R[node][dst];
                    next_node = node_port_dst[node][out_port]
                    path.append(next_node)
                    node = next_node
                MatrixPath[src][dst] = path
        return (MatrixPath)

    def _get_graph_for_tarfile(self, tar):
        """
        

        Parameters
        ----------
        tar : str
            tar file where the graph file is located.

        Returns
        -------
        ret : graph
            Graph representation of the network.

        """
        
        for member in tar.getmembers():
            if 'graph' in member.name:
                f = tar.extractfile(member)
                ret = networkx.read_gml(f, destringizer=int)
                return ret
        
    def __process_params_file(self,params_file):
        simParameters = {}
        for line in params_file:
            line = line.decode()
            if ("simulationDuration" in line):
                ptr = line.find("=")
                simulation_time = int (line[ptr+1:])
                simParameters["simulationTime"] = simulation_time
                continue
            if ("lowerLambda" in line):
                ptr = line.find("=")
                lowerLambda = float(line[ptr+1:])
                simParameters["lowerLambda"] = lowerLambda
                continue
            if ("upperLambda" in line):
                ptr = line.find("=")
                upperLambda = float(line[ptr+1:])
                simParameters["upperLambda"] = upperLambda
        return(simParameters)
    
    def __process_graph(self,G):
        netSize = G.number_of_nodes()
        for src in range(netSize):
            for dst in range(netSize):
                if not dst in G[src]:
                    continue
                bw = G[src][dst][0]['bandwidth']
                bw = bw.replace("kbps","000")
                G[src][dst][0]['bandwidth'] = bw
                
    def __iter__(self):
        """
        

        Yields
        ------
        s : Sample
            Sample instance containing information about the last line read
            from the dataset.

        """
        
        g = None
        
        tuple_files = []
        graphs_dic = {}
        for root, dirs, files in os.walk(self.data_folder):
            if (not "graph_attr.txt" in files):
                continue
            # Generate graphs dictionaries
            graphs_dic[root] = networkx.read_gml(os.path.join(root, "graph_attr.txt"), destringizer=int)
            self.__process_graph(graphs_dic[root])
            # Extend the list of files to process
            tuple_files.extend([(root, f) for f in files if f.endswith("tar.gz")])

        if self.shuffle:
            random.Random(1234).shuffle(tuple_files)
            
            
        for root, file in tuple_files:
            g = graphs_dic[root]
            
            tar = tarfile.open(os.path.join(root, file), 'r:gz')
            dir_info = tar.next()
            routing_file = tar.extractfile(dir_info.name+"/Routing.txt")
            results_file = tar.extractfile(dir_info.name+"/simulationResults.txt")
            if (dir_info.name+"/flowSimulationResults.txt" in tar.getnames()):
                flowresults_file = tar.extractfile(dir_info.name+"/flowSimulationResults.txt")
            else:
                flowresults_file = None
            params_file = tar.extractfile(dir_info.name+"/params.ini")
            simParameters = self.__process_params_file(params_file)
            
            routing_matrix= self._create_routing_matrix(g, routing_file)
            while(True):
                s = Sample()
                s._set_topology_object(g)
                s._set_data_set_file_name(os.path.join(root, file))
                
                s._results_line = results_file.readline().decode()[:-2]
                if (flowresults_file):
                    s._flowresults_line = flowresults_file.readline().decode()[:-2]
                else:
                    s._flowresults_line = None
                
                if (len(s._results_line) == 0):
                    break
                    
                self._process_flow_results_traffic_line(s._results_line, s._flowresults_line, simParameters, s)
                s._set_routing_matrix(routing_matrix)
                
                yield s
    
    def _process_flow_results_traffic_line(self, rline, fline, simParameters, s):
        """
        

        Parameters
        ----------
        rline : str
            Last line read in the results file.
        fline : str
            Last line read in the flows file.
        s : Sample
            Instance of Sample associated with the current iteration.

        Returns
        -------
        None.

        """
        
        sim_time = simParameters["simulationTime"]
        r = rline.split(',')
        if (fline):
            f = fline.split(',')
        else:
            f = r

        maxAvgLambda = 0
        
        m_result = []
        m_traffic = []
        netSize = s.get_network_size()
        globalPackets = 0
        globalLosses = 0
        globalDelay = 0
        offset = netSize*netSize*3
        for src_node in range (netSize):
            new_result_row = []
            new_traffic_row = []
            for dst_node in range (netSize):
                offset_t = (src_node * netSize + dst_node)*3
                offset_d = offset + (src_node * netSize + dst_node)*8
                pcktsGen = float(r[offset_t + 1])
                pcktsDrop = float(r[offset_t + 2])
                pcktsDelay = float(r[offset_d])
                dict_result_agg = {
                    'PktsDrop':numpy.round(pcktsDrop/sim_time,6),
                    "AvgDelay":pcktsDelay,
                    "AvgLnDelay":float(r[offset_d + 1]),
                    "p10":float(r[offset_d + 2]), 
                    "p20":float(r[offset_d + 3]), 
                    "p50":float(r[offset_d + 4]), 
                    "p80":float(r[offset_d + 5]), 
                    "p90":float(r[offset_d + 6]), 
                    "Jitter":float(r[offset_d + 7])}
                
                if (src_node != dst_node):
                    globalPackets += pcktsGen
                    globalLosses += pcktsDrop
                    globalDelay += pcktsDelay
                
                lst_result_flows = []
                lst_traffic_flows = []
                
                dict_result_tmp = {
                    'PktsDrop':numpy.round(pcktsDrop/sim_time,6),
                    "AvgDelay":pcktsDelay,
                    "AvgLnDelay":float(r[offset_d + 1]),
                    "p10":float(r[offset_d + 2]), 
                    "p20":float(r[offset_d + 3]), 
                    "p50":float(r[offset_d + 4]), 
                    "p80":float(r[offset_d + 5]), 
                    "p90":float(r[offset_d + 6]), 
                    "Jitter":float(r[offset_d + 7])}
                lst_result_flows.append(dict_result_tmp)
                
                
                dict_traffic = {}
                dict_traffic['AvgBw'] = float(r[offset_t])*1000
                dict_traffic['PktsGen'] = numpy.round(pcktsGen/sim_time,6)
                dict_traffic['TotalPktsGen'] = float(pcktsGen)
                dict_traffic['ToS'] = 0
                self._timedistparams(dict_traffic)
                self._sizedistparams(dict_traffic)
                lst_traffic_flows.append (dict_traffic)
                
                # From kbps to bps
                dict_traffic_agg = {'AvgBw':float(r[offset_t])*1000,
                                    'PktsGen':numpy.round(pcktsGen/sim_time,6),
                                    'TotalPktsGen':pcktsGen}

                dict_result_srcdst = {}                           
                dict_traffic_srcdst = {}

                dict_result_srcdst['AggInfo'] = dict_result_agg
                dict_result_srcdst['Flows'] = lst_result_flows
                dict_traffic_srcdst['AggInfo'] = dict_traffic_agg
                dict_traffic_srcdst['Flows'] = lst_traffic_flows
                new_result_row.append(dict_result_srcdst)
                new_traffic_row.append(dict_traffic_srcdst)
                
            m_result.append(new_result_row)
            m_traffic.append(new_traffic_row)
        m_result = numpy.asmatrix(m_result)
        m_traffic = numpy.asmatrix(m_traffic)
        s._set_performance_matrix(m_result)
        s._set_traffic_matrix(m_traffic)
        s._set_global_packets(numpy.round(globalPackets/sim_time,6))
        s._set_global_losses(numpy.round(globalLosses/sim_time,6))
        s._set_global_delay(globalDelay/(netSize*(netSize-1)))
        

    # Dataset v0 only contain exponential traffic with avg packet size of 1000
    def _timedistparams(self, dict_traffic):
        """
        

        Parameters
        ----------
        dict_traffic: dictionary
            Dictionary to fill with the time distribution information
            extracted from data
        

        """
        
        dict_traffic['TimeDist'] = TimeDist.EXPONENTIAL_T
        params = {}
        params['EqLambda'] = dict_traffic['AvgBw']
        params['AvgPktsLambda'] = dict_traffic['AvgBw'] / 1000 # Avg Pkt size 1000
        params['ExpMaxFactor'] = 10
        dict_traffic['TimeDistParams'] = params
        
    
    # Dataset v0 only contains binomial traffic with avg packet size of 1000
    def _sizedistparams(self, dict_traffic):
        """
        

        Parameters
        ----------
        dict_traffic : dictionary
            Dictionary to fill with the size distribution information
            extracted from data

        """
        dict_traffic['SizeDist'] = SizeDist.BINOMIAL_S
        params = {}
        params['AvgPktSize'] = 1000
        params['PktSize1'] = 300
        params['PktSize2'] = 1700
        dict_traffic['SizeDistParams'] = params

