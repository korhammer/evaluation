__author__ = 'korhammer'

import pandas as pd
import h5py
import numpy as np

from os import listdir
from os.path import join, isfile, isdir

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import misc


class Evaluation:
    def __init__(self, allocate=50000):
        self.results = pd.DataFrame()
        self.filtered = self.results
        self.order = {}
        self.order_all = {}
        self.add_counter = -1
        self.allocate = allocate
        self.patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

    def add_folder(self, path, file_method, recursive=True):
        """ Add a folder with files of a certain file type. """

        for file in listdir(path):
            full_file_path = join(path, file)
            if isfile(full_file_path):
                self.add_hdf5(full_file_path)
            elif isdir(full_file_path) and recursive:
                self.add_folder(full_file_path, file_method, recursive)

    def add_hdf5(self, path):
        """ Add a HDF5 file. """
        if path.endswith('.hdf5'):
            h5File = h5py.File(path, 'r')
            for group in h5File:
                hdf_dic = {}
                for att in h5File[group].attrs:
                    hdf_dic[att] = h5File[group].attrs[att]
                hdf_dic['train mean'] = np.mean(h5File[group + '/LL train'])
                hdf_dic['train std'] = np.std(h5File[group + '/LL train'])
                hdf_dic['test mean'] = np.mean(h5File[group + '/LL test'])
                hdf_dic['test std'] = np.std(h5File[group + '/LL test'])

                # allocate large DataFrame and initialize with nans
                if self.add_counter < 0:
                    self.results = self.results.append(hdf_dic, ignore_index=True)
                    self.results = pd.DataFrame(np.zeros([self.allocate, len(self.results.columns)]) + np.nan,
                                                columns=self.results.columns)
                    self.add_counter = 0

                self.results.iloc[self.add_counter] = pd.Series(hdf_dic)
                self.add_counter += 1

            h5File.close()

    def set_order(self, attribute, order):
        """
        Set the order for a certain attribute to either ascending, descending, or a complete given order.
        """
        if not (set(self.results[attribute].unique()).issubset(order) or isinstance(order, str)):
            raise InputError('order', order, 'is inconsistent with current entries')
        self.order[attribute] = order

    def set_all_orders(self):
        for att in self.results.columns:
            if self.order.has_key(att):
                if isinstance(self.order[att], list):
                    self.order_all[att] = self.order[att]
                elif self.order[att] is 'ascend':
                    self.order_all[att] = np.sort(self.results[att].unique())
                elif self.order[att] is 'descend':
                    self.order_all[att] = np.sort(self.results[att].unique())[::-1]
            else:
                self.order_all[att] = self.results[att].unique()

    def filter(self, attribute, values, filter_type='in'):
        if filter_type is 'in' or 'is':
            if not isinstance(values, list):
                values = [values]
            self.filtered = self.filtered[self.filtered[attribute].isin(values)]
        elif filter_type is ('<' or 'smaller'):
            self.filtered = self.filtered[self.filtered[attribute] < values]
        elif filter_type is ('>' or 'greater'):
            self.filtered = self.filtered[self.filtered[attribute] > values]
        elif filter_type is ('<=' or 'se'):
            self.filtered = self.filtered[self.filtered[attribute] <= values]
        elif filter_type is ('>=' or 'ge'):
            self.filtered = self.filtered[self.filtered[attribute] >= values]
        elif filter_type is 'not':
            if not isinstance(values, list):
                values = [values]
            self.filtered = self.filtered[-self.filtered[attribute].isin(values)]
        else:
            warnings.warn('Filter type unknown. No filter was applied.', UserWarning)

    def unfilter(self):
        self.filtered = self.results

    def convert_flags(self, flags=['rectified', 'scaled', 'whitened'], name='flags', default='raw'):
        self.results.loc[:, name] = ''
        try:
            for flag in flags:
                self.results[name] += self.results[flag] * flag
        except:
            for flag in flags:
                single_flag = self.results[flag].copy()
                single_flag[single_flag != 0] = flag
                single_flag[single_flag == 0] = ''
                self.results.loc[:, name] += single_flag
        single_flag = self.results[name].copy()
        single_flag[single_flag == ''] = default
        self.results.loc[:, name] = single_flag

    def convert_flags_abbr(self, flags=['rectified', 'scaled', 'whitened'], name='flags', default='raw'):
        self.results.loc[:, name] = ''

        try:
            for flag in flags:
                self.results[name] += self.results[flag] * flag[0]
        except:
            for flag in flags:
                single_flag = self.results[flag].copy()
                single_flag[single_flag != 0] = flag[0]
                single_flag[single_flag == 0] = ''
                self.results.loc[:, name] += single_flag
        single_flag = self.results[name].copy()
        single_flag[single_flag == ''] = default
        self.results.loc[:, name] = single_flag


    def best_results_for(self, attributes, objective='test mean',
                         outputs=['test mean', 'train mean', 'test std', 'train std'],
                         fun='max'):
        if fun == 'max':
            best = self.filtered.sort(objective).groupby(attributes)[outputs].last()
        elif fun == 'min':
            best = self.filtered.sort(objective).groupby(attributes)[outputs].first()
        elif fun == 'mean':
            best = self.filtered.groupby(attributes)[outputs].mean()
        elif fun == 'count':
            best = self.filtered.groupby(attributes)[objective].count()
        return best

    def make_same(self, attribute, values):
        self.results[attribute].replace(values, values[0], inplace=True)

    def rename_attribute(self, attribute, new_name):
        self.results.rename(columns={attribute: new_name}, inplace=True)

    def bring_in_order(self, attributes, attribute):
        satts = set(attributes)
        return [(i, att) for i, att in enumerate(self.order_all[attribute]) if att in satts]

    def group_subplots(self, best, counts=None,
                       error=False, no_rows=2,
                       adapt_bottom=True, plot_range=None, base=5, eps=.5,

                       plot_fit=True, colormap='pastel1', max_n_cols=10,
                       legend_position='lower right', legend_pad='not implemented',
                       print_value='auto'):
        """ Create a single barplot for each group of the first attribute in best.
        """
        no_subplots = len(best.index.levels[0])
        f, ax_arr = plt.subplots(no_rows, np.int(np.ceil(no_subplots / np.float(no_rows))))

        ax_flat = ax_arr.flatten()

        att_names = best.index.names
        self.set_all_orders()

        lev0 = self.bring_in_order(best.index.levels[0], att_names[0])
        lev1 = self.bring_in_order(best.index.levels[1], att_names[1])

        best = best.reset_index()
        if counts is not None:
            counts = counts.reset_index()
        bar_x = np.arange(len(lev1))
        offset = 1

        cmap = plt.cm.get_cmap(colormap)
        dummy_artists = []

        for plt_i, (lev0_ind, lev0_att) in enumerate(lev0):
            for bar_i, (lev1_ind, lev1_att) in enumerate(lev1):

                c = cmap(np.float(lev1_ind) / len(lev1))
                dummy_artists.append(Rectangle((0, 0), 1, 1, fc=c))

                # compute plot limits
                if plot_range:
                    bottom = plot_range[0]
                    ceil = plot_range[1]
                elif adapt_bottom:
                    relevant = best[(best[att_names[0]] == lev0_att) & -(best['test mean'] == 0)]
                    if error:
                        ceil = misc.based_ceil(np.max(relevant['test mean'])
                                               + np.max(relevant['test std']) + eps, base)
                        bottom = misc.based_floor(np.min(relevant['test mean'])
                                                  - np.max(relevant['test std']) - eps, base)
                    else:
                        ceil = misc.based_ceil(np.max(relevant['test mean']) + eps, base)
                        bottom = misc.based_floor(np.min(relevant['test mean']) - eps, base)

                test_mean = misc.float(best[(best[att_names[0]] == lev0_att)
                                            & (best[att_names[1]] == lev1_att)]['test mean'])
                test_std = misc.float(best[(best[att_names[0]] == lev0_att)
                                           & (best[att_names[1]] == lev1_att)]['test std'])
                train_mean = misc.float(best[(best[att_names[0]] == lev0_att)
                                             & (best[att_names[1]] == lev1_att)]['train mean'])
                train_std = misc.float(best[(best[att_names[0]] == lev0_att)
                                            & (best[att_names[1]] == lev1_att)]['train std'])

                # create bar plots
                if (test_mean is not 0) and (test_mean is not np.nan) and (train_mean is not np.nan):
                    if plot_fit:
                        if error:
                            ax_flat[plt_i].bar(bar_x[bar_i], train_mean - bottom, .4,
                                               color=c, bottom=bottom, yerr=train_std, ecolor='gray', alpha=.5,
                                               linewidth=0.)
                            ax_flat[plt_i].bar(bar_x[bar_i] + .4, test_mean - bottom, .4,
                                               color=c, bottom=bottom, yerr=test_std, ecolor='gray', linewidth=0.)
                        else:
                            ax_flat[plt_i].bar(bar_x[bar_i], train_mean - bottom, .4,
                                               color=c, bottom=bottom, alpha=.5, linewidth=0.)
                            ax_flat[plt_i].bar(bar_x[bar_i] + .4, test_mean - bottom, .4,
                                               color=c, bottom=bottom, linewidth=0.)

                        if print_value is True or (print_value is not False and counts is None):
                            ax_flat[plt_i].text(bar_x[bar_i] + .25, (test_mean + bottom) / 2, '%.2f' % train_mean,
                                            ha='center', va='top', rotation='vertical')

                    else:
                        if error:
                            ax_flat[plt_i].bar(bar_x[bar_i], test_mean - bottom,
                                               color=c, bottom=bottom, yerr=test_std, ecolor='gray', linewidth=0.)
                        else:
                            ax_flat[plt_i].bar(bar_x[bar_i], test_mean - bottom,
                                               color=c, bottom=bottom, linewidth=0.)

                        if print_value is True or (print_value is not False and counts is None):
                            ax_flat[plt_i].text(bar_x[bar_i] + .5, (test_mean + bottom) / 2, '%.2f' % test_mean,
                                            ha='center', va='center', rotation='vertical')


                    # print count
                    if counts is not None:
                        count = misc.int(counts[(counts[att_names[0]] == lev0_att)
                                              & (counts[att_names[1]] == lev1_att)]['test mean'])

                        if count > 0:
                            ax_flat[plt_i].text(bar_x[bar_i] + .4, (test_mean + bottom) / 2, '%d' % count,
                                            ha='center', va='center', rotation='vertical')



                ax_flat[plt_i].set_title(lev0_att)
                ax_flat[plt_i].set_xticks([])
                ax_flat[plt_i].set_ylim(bottom, ceil)

                ax_flat[plt_i].spines['top'].set_visible(False)
                ax_flat[plt_i].spines['right'].set_visible(False)
                ax_flat[plt_i].spines['left'].set_color('gray')
                ax_flat[plt_i].spines['bottom'].set_color('gray')

        #for plt_i in range(len(lev0), len(ax_flat)): #wrong range, doesn't do anything, why turn off anyway?
        #    ax_flat[plt_i].axis('off')

        legend = [(int(att) if isinstance(att, float) else att) for i, att in lev1]

        n_col=len(legend)
        if n_col>max_n_cols:
            n_col=int((n_col+1)/2)

        plt.figlegend(dummy_artists, legend, loc=legend_position, ncol=n_col, title=att_names[1])

    def group_subplots_3(self, best, counts=None,
                       error=False, no_rows=2,
                       adapt_bottom=True, plot_range=None, base=5, eps=.5,

                       plot_fit=True, colormap='pastel1', max_n_cols=10,
                       legend_position='lower right', legend_pad='not implemented',
                       print_value='auto'):
        """ Create a single barplot for each group of the first attribute in best.
        """
        no_subplots = len(best.index.levels[0])
        f, ax_arr = plt.subplots(no_rows, np.int(np.ceil(no_subplots / np.float(no_rows))))

        ax_flat = ax_arr.flatten()

        att_names = best.index.names
        self.set_all_orders()

        lev0 = self.bring_in_order(best.index.levels[0], att_names[0])
        lev1 = self.bring_in_order(best.index.levels[1], att_names[1])
        lev2 = self.bring_in_order(best.index.levels[2], att_names[2])

        best = best.reset_index()
        if counts is not None:
            counts = counts.reset_index()
        bar_x = np.arange(len(lev1))
        block_x=np.arange(len(lev2))
        width=1.0/len(lev2)
        bar_width=.4*width
        offset = 1

        cmap = plt.cm.get_cmap(colormap)
        dummy_artists = []

        for plt_i, (lev0_ind, lev0_att) in enumerate(lev0):
            for bar_i, (lev1_ind, lev1_att) in enumerate(lev1):
                c = cmap(np.float(lev1_ind) / len(lev1))
                dummy_artists.append(Rectangle((0, 0), 1, 1, fc=c))
                for block_i, (lev2_ind, lev2_att) in enumerate(lev2):
                    h=self.patterns[block_i]

                    # compute plot limits
                    if plot_range:
                        bottom = plot_range[0]
                        ceil = plot_range[1]
                    elif adapt_bottom:
                        relevant = best[(best[att_names[0]] == lev0_att) & -(best['test mean'] == 0)]
                        if error:
                            ceil = misc.based_ceil(np.max(relevant['test mean'])
                                                   + np.max(relevant['test std']) + eps, base)
                            bottom = misc.based_floor(np.min(relevant['test mean'])
                                                      - np.max(relevant['test std']) - eps, base)
                        else:
                            ceil = misc.based_ceil(np.max(relevant['test mean']) + eps, base)
                            bottom = misc.based_floor(np.min(relevant['test mean']) - eps, base)

                    test_mean = misc.float(best[(best[att_names[0]] == lev0_att)
                                                & (best[att_names[1]] == lev1_att)
                                                & (best[att_names[2]] == lev2_att)]['test mean'])
                    test_std = misc.float(best[(best[att_names[0]] == lev0_att)
                                               & (best[att_names[1]] == lev1_att)
                                               & (best[att_names[2]] == lev2_att)]['test std'])
                    train_mean = misc.float(best[(best[att_names[0]] == lev0_att)
                                                 & (best[att_names[1]] == lev1_att)
                                                 & (best[att_names[2]] == lev2_att)]['train mean'])
                    train_std = misc.float(best[(best[att_names[0]] == lev0_att)
                                                & (best[att_names[1]] == lev1_att)
                                                & (best[att_names[2]] == lev2_att)]['train std'])

                    # create bar plots
                    if (test_mean is not 0) and (test_mean is not np.nan) and (train_mean is not np.nan):
                        if plot_fit:
                            if error:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, train_mean - bottom, bar_width,
                                                   color=c, hatch=h, bottom=bottom, yerr=train_std, ecolor='gray', alpha=.5,
                                                   linewidth=0.)
                                ax_flat[plt_i].bar(bar_x[bar_i] +block_x[block_i]*width+ bar_width, test_mean - bottom, bar_width,
                                                   color=c, hatch=h, bottom=bottom, yerr=test_std, ecolor='gray', linewidth=0.)
                            else:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, train_mean - bottom, bar_width,
                                                   color=c, hatch=h, bottom=bottom, alpha=.5, linewidth=0.)
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width + bar_width, test_mean - bottom, bar_width,
                                                   color=c, hatch=h, bottom=bottom, linewidth=0.)

                            if print_value is True or (print_value is not False and counts is None):
                                ax_flat[plt_i].text(bar_x[bar_i]+block_x[block_i]*width + .25, (test_mean + bottom) / 2, '%.2f' % train_mean,
                                                ha='center', va='top', rotation='vertical')

                        else:
                            if error:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, test_mean - bottom,
                                                   color=c, hatch=h, bottom=bottom, yerr=test_std, ecolor='gray', linewidth=0.)
                            else:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, test_mean - bottom,
                                                   color=c, hatch=h, bottom=bottom, linewidth=0.)

                            if print_value is True or (print_value is not False and counts is None):
                                ax_flat[plt_i].text(bar_x[bar_i]+block_x[block_i]*width + .5, (test_mean + bottom) / 2, '%.2f' % test_mean,
                                                ha='center', va='center', rotation='vertical')


                        # print count
                        if counts is not None:
                            count = misc.int(counts[(counts[att_names[0]] == lev0_att)
                                                  & (counts[att_names[1]] == lev1_att)
                                                  & (counts[att_names[2]] == lev2_att)]['test mean'])

                            if count > 0:
                                ax_flat[plt_i].text(bar_x[bar_i] +block_x[block_i]*bar_width+ .4, (test_mean + bottom) / 2, '%d' % count,
                                                ha='center', va='center', rotation='vertical')



                    ax_flat[plt_i].set_title(lev0_att)
                    ax_flat[plt_i].set_xticks([])
                    ax_flat[plt_i].set_ylim(bottom, ceil)

                    ax_flat[plt_i].spines['top'].set_visible(False)
                    ax_flat[plt_i].spines['right'].set_visible(False)
                    ax_flat[plt_i].spines['left'].set_color('gray')
                    ax_flat[plt_i].spines['bottom'].set_color('gray')

            for block_i, (lev2_ind, lev2_att) in enumerate(lev2):
                h=self.patterns[block_i]
                dummy_artists.append(Rectangle((0, 0), 1, 1, fc='w', hatch=h))
        #for plt_i in range(len(lev0), len(ax_flat)):  #doesn't do anything, why turn off anyway?
        #    ax_flat[plt_i].axis('off')

        legend = [(int(att) if isinstance(att, float) else att) for i, att in lev1]+[(int(att) if isinstance(att, float) else att) for i, att in lev2]

        n_col=len(legend)
        if n_col>max_n_cols:
            n_col=int((n_col+1)/2)

        plt.figlegend(dummy_artists, legend, loc=legend_position, ncol=n_col, title=att_names[1])

    def group_subplots_all_parameters(self, best, counts=None,
                       error=False, no_rows=2,
                       adapt_bottom=True, plot_range=None, base=5, eps=.5,

                       plot_fit=True, colormap='pastel1', max_n_cols=10,
                       legend_position='lower right', legend_pad='not implemented',
                       print_value='auto'):
        """ Create a single barplot for each group of the first attribute in best.
        """
        no_subplots = len(best.index.levels[0])
        f, ax_arr = plt.subplots(no_rows, np.int(np.ceil(no_subplots / np.float(no_rows))))

        ax_flat = ax_arr.flatten()

        att_names = best.index.names
        self.set_all_orders()

        lev0 = self.bring_in_order(best.index.levels[0], att_names[0])
        lev1 = self.bring_in_order(best.index.levels[1], att_names[1])
        lev2 = self.bring_in_order(best.index.levels[2], att_names[2])

        best = best.reset_index()
        if counts is not None:
            counts = counts.reset_index()
        bar_x = np.arange(len(lev1))
        block_x=np.arange(len(lev2))
        width=1.0/len(lev2)
        bar_width=.4*width
        offset = 1

        cmap = plt.cm.get_cmap(colormap)
        dummy_artists = []
        ticks=[]
        tick_labels=[]

        for plt_i, (lev0_ind, lev0_att) in enumerate(lev0):
            for bar_i, (lev1_ind, lev1_att) in enumerate(lev1):
                c = cmap(np.float(lev1_ind) / len(lev1))
                dummy_artists.append(Rectangle((0, 0), 1, 1, fc=c))
                for block_i, (lev2_ind, lev2_att) in enumerate(lev2):
                    # compute plot limits
                    if plot_range:
                        bottom = plot_range[0]
                        ceil = plot_range[1]
                    elif adapt_bottom:
                        relevant = best[(best[att_names[0]] == lev0_att) & -(best['test mean'] == 0)]
                        if error:
                            ceil = misc.based_ceil(np.max(relevant['test mean'])
                                                   + np.max(relevant['test std']) + eps, base)
                            bottom = misc.based_floor(np.min(relevant['test mean'])
                                                      - np.max(relevant['test std']) - eps, base)
                        else:
                            ceil = misc.based_ceil(np.max(relevant['test mean']) + eps, base)
                            bottom = misc.based_floor(np.min(relevant['test mean']) - eps, base)

                    test_mean = misc.float(best[(best[att_names[0]] == lev0_att)
                                                & (best[att_names[1]] == lev1_att)
                                                & (best[att_names[2]] == lev2_att)]['test mean'])
                    test_std = misc.float(best[(best[att_names[0]] == lev0_att)
                                               & (best[att_names[1]] == lev1_att)
                                               & (best[att_names[2]] == lev2_att)]['test std'])
                    train_mean = misc.float(best[(best[att_names[0]] == lev0_att)
                                                 & (best[att_names[1]] == lev1_att)
                                                 & (best[att_names[2]] == lev2_att)]['train mean'])
                    train_std = misc.float(best[(best[att_names[0]] == lev0_att)
                                                & (best[att_names[1]] == lev1_att)
                                                & (best[att_names[2]] == lev2_att)]['train std'])

                    # create bar plots
                    if (test_mean is not 0) and (test_mean is not np.nan) and (train_mean is not np.nan):
                        if plot_fit:
                            if error:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, train_mean - bottom, bar_width,
                                                   color=c, bottom=bottom, yerr=train_std, ecolor='gray', alpha=.5,
                                                   linewidth=0.)
                                ax_flat[plt_i].bar(bar_x[bar_i] +block_x[block_i]*width+ bar_width, test_mean - bottom, bar_width,
                                                   color=c, bottom=bottom, yerr=test_std, ecolor='gray', linewidth=0.)
                            else:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, train_mean - bottom, bar_width,
                                                   color=c, bottom=bottom, alpha=.5, linewidth=0.)
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width + bar_width, test_mean - bottom, bar_width,
                                                   color=c, bottom=bottom, linewidth=0.)

                            if print_value is True or (print_value is not False and counts is None):
                                ax_flat[plt_i].text(bar_x[bar_i]+block_x[block_i]*width + .25, (test_mean + bottom) / 2, '%.2f' % train_mean,
                                                ha='center', va='top', rotation='vertical')

                        else:
                            if error:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, test_mean - bottom,
                                                   color=c, bottom=bottom, yerr=test_std, ecolor='gray', linewidth=0.)
                            else:
                                ax_flat[plt_i].bar(bar_x[bar_i]+block_x[block_i]*width, test_mean - bottom,
                                                   color=c, bottom=bottom, linewidth=0.)

                            if print_value is True or (print_value is not False and counts is None):
                                ax_flat[plt_i].text(bar_x[bar_i]+block_x[block_i]*width + .5, (test_mean + bottom) / 2, '%.2f' % test_mean,
                                                ha='center', va='center', rotation='vertical', hatch=self.patterns[block_i])

                        if plt_i==0:
                            ticks.append(bar_x[bar_i]+block_x[block_i]*width+width*0.5)
                        tick_labels+=[lev2_att]
                        # print count
                        if counts is not None:
                            count = misc.int(counts[(counts[att_names[0]] == lev0_att)
                                                  & (counts[att_names[1]] == lev1_att)
                                                  & (counts[att_names[2]] == lev2_att)]['test mean'])

                            if count > 0:
                                ax_flat[plt_i].text(bar_x[bar_i] +block_x[block_i]*bar_width+ .4, (test_mean + bottom) / 2, '%d' % count,
                                                ha='center', va='center', rotation='vertical')



                    ax_flat[plt_i].set_title(lev0_att)
                    ax_flat[plt_i].set_xticks([])
                    ax_flat[plt_i].set_ylim(bottom, ceil)

                    ax_flat[plt_i].spines['top'].set_visible(False)
                    ax_flat[plt_i].spines['right'].set_visible(False)
                    ax_flat[plt_i].spines['left'].set_color('gray')
                    ax_flat[plt_i].spines['bottom'].set_color('gray')

        for plt_i in range(len(ax_flat)):
            #ax_flat[plt_i].axis('off')
            ax_flat[plt_i].set_xticks( ticks )
            ax_flat[plt_i].set_xticklabels( tick_labels, rotation=90 )
        print 'shit'
        legend = [(int(att) if isinstance(att, float) else att) for i, att in lev1]

        n_col=len(legend)
        if n_col>max_n_cols:
            n_col=int((n_col+1)/2)

        plt.figlegend(dummy_artists, legend, loc=legend_position, ncol=n_col, title=att_names[1])

