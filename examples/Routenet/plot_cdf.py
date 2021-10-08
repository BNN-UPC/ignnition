"""
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


def main():
    gbn = list()
    geant2 = list()
    
    with open('GBN.pkl', 'rb') as f:
        gbn = pickle.load(f)
    
    with open('Geant2.pkl', 'rb') as f:
        geant2 = pickle.load(f)

    gbn = gbn[:1000]
    geant2 = geant2[:1000]

    fig, ax = plt.subplots()
    n = np.arange(1,len(gbn)+1) / np.float(len(gbn))
    Xs = np.sort(gbn)
    ax.step(Xs,n, c='darkorange', linestyle='-.', label="GBN", linewidth=3) 
    n = np.arange(1,len(gbn)+1) / np.float(len(geant2))
    Xs = np.sort(geant2)
    ax.step(Xs,n, c='green', linestyle='-', label="GEANT2", linewidth=3) 

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['pdf.fonttype'] = 42
    plt.ylim((0, 1.001))
    plt.xlim((-0.4, 0.4001))
    plt.xticks(np.arange(-0.4, 0.4001, 0.1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('CDF', fontsize=17)
    plt.xlabel("Relative Error (%)", fontsize=20)
    plt.legend(prop={'size': 16, 'weight': 'bold'}, loc='lower right')
    plt.grid(color='gray')
    lgd = plt.legend(loc="lower left", bbox_to_anchor=(0.0, 0.8), ncol=1)
    plt.tight_layout()
    #plt.show()
    plt.savefig('CDF_GEANT_GBN.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.clf()


if __name__ == "__main__":
    main()
