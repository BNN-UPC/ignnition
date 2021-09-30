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
    
    # with open('GEANT2.pkl', 'rb') as f:
    #     geant2 = pickle.load(f)

    fig, ax = plt.subplots()
    n = np.arange(1,len(gbn)+1) / np.float(len(gbn))
    Xs = np.sort(gbn)
    ax.step(Xs,n, c='darkorange', linestyle='--', label="GBN", linewidth=3) 

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['pdf.fonttype'] = 42
    plt.ylim((0, 1.005))
    plt.xlim((-0.1, 0.25))
    plt.xticks(np.arange(-0.1, 0.25, 0.05))
    plt.ylabel('CDF', fontsize=17)
    plt.xlabel("Mean Relative Error", fontsize=20)
    plt.legend(prop={'size': 16, 'weight': 'bold'}, loc='lower right')
    plt.grid(color='gray')
    lgd = plt.legend(loc="lower left", bbox_to_anchor=(0.1, 0.8), ncol=2)
    plt.tight_layout()
    #plt.show()
    plt.savefig('CDF_GEANT_GBN.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.clf()


if __name__ == "__main__":
    main()
