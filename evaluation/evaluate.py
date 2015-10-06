__author__ = 'korhammer'

import pandas as pd
import h5py
import numpy as np

from os import listdir
from os.path import join, isfile, isdir

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import misc
import itertools as it

import warnings

def _flagify(dat, flags, delimiter):
    return delimiter.join([flag for flag in flags if dat[flag]])


class Evaluation:
    """
    This class helps in creating meaningful plots to help interpret the
    results of a series of experiments with different parameters.
    """

    def __init__(self):
        self.results = pd.DataFrame()
        self._fltrd = self.results
        self._att_order_demand = {}
        self._att_order = {}
        self.patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        self._color_cycl_f = {}

    def set_order(self, attribute, order):
        """
        Set the order for a certain attribute to either ascending,
        descending, or a complete given order. Plots will be in this order.
        """
        if not (set(self.results[attribute].unique()).issubset(order)
                or isinstance(order, str)):
            raise InputError('order', order,
                             'is inconsistent with current entries')
        self._att_order_demand[attribute] = order

    def set_color_cycling_factor(self, attribute, factor=1.):
        self._color_cycl_f[attribute] = np.float(factor)

    def _process_demanded_att_order(self):
        """
        Process the demanded attribute orders for all attributes.
        """
        for att in self.results.columns:
            if self._att_order_demand.has_key(att):
                if isinstance(self._att_order_demand[att], list):
                    self._att_order[att] = [val for val in self._att_order_demand[att]
                                            if val in self.results[att].unique()]
                    #self._att_order[att] = self._att_order_demand[att]
                elif self._att_order_demand[att].startswith('ascend'):
                    self._att_order[att] = list(np.sort(self.results[att].unique()))
                elif self._att_order_demand[att].startswith('descend'):
                    self._att_order[att] = list(np.sort(
                        self.results[att].unique()
                    )[::-1])
                else:
                    warnings.warn(
                        'Attribute order %s for attribute %s unknown.'
                        ' Please choose one of ascend, descend or a'
                        ' list of attributes.'
                        % (self._att_order_demand[att], att))
            else:
                self._att_order[att] = self.results[att].unique()

    def filter(self, attribute, values, filter_type='in'):
        """
        Filter
        """
        if filter_type is 'in' or 'is':
            if not isinstance(values, list):
                values = [values]
            self._fltrd = self._fltrd[self._fltrd[attribute].isin(values)]
        elif filter_type is ('<' or 'smaller'):
            self._fltrd = self._fltrd[self._fltrd[attribute] < values]
        elif filter_type is ('>' or 'greater'):
            self._fltrd = self._fltrd[self._fltrd[attribute] > values]
        elif filter_type is ('<=' or 'se'):
            self._fltrd = self._fltrd[self._fltrd[attribute] <= values]
        elif filter_type is ('>=' or 'ge'):
            self._fltrd = self._fltrd[self._fltrd[attribute] >= values]
        elif filter_type is 'not':
            if not isinstance(values, list):
                values = [values]
            self._fltrd = self._fltrd[-self._fltrd[attribute].isin(values)]
        else:
            warnings.warn('Filter type unknown. No filter was applied.',
                          UserWarning)

    def unfilter(self):
        self._fltrd = self.results

    def convert_flags(self, flags, name='flags', all_unset='raw', delimiter=''):
        self.results[name] = self.results.apply(_flagify,
                                                   flags=flags,
                                                   delimiter=delimiter,
                                                   axis=1)
        self.results.loc[self.results[name]=='', name] = all_unset

        self.set_order(name,
                       [all_unset] +
                       [' '.join([val for val in vals if len(val)>0])
                        for vals in it.product(*[['', flag] for flag in flags])])

    def best_results_for(self, attributes,
                         objective='test mean',
                         outputs=['test mean', 'train mean',
                                  'test std', 'train std'],
                         fun='max'):

        grouped = self._fltrd.sort(objective).groupby(attributes)[outputs]

        if fun == 'max':
            best = grouped.last()
        elif fun == 'min':
            best = grouped.first()
        elif fun == 'mean':
            best = grouped.mean()
        elif fun == 'count':
            best = grouped.count()
        return best

    def make_same(self, attribute, values):
        self.results[attribute].replace(values, values[0], inplace=True)

    def rename_attribute(self, attribute, new_name):
        self.results.rename(columns={attribute: new_name}, inplace=True)

    def _bring_in_order(self, attributes, attribute):
        satts = set(attributes)
        return [(i, att)
                for i, att in enumerate(self._att_order[attribute])
                if att in satts]

    def group_subplots(self, best, counts=None,
                       error=False, no_rows=2,
                       adapt_bottom=True, plot_range=None, base=.1, eps=.05,
                       plot_fit=True, cmap='Pastel1', legend_ncol=10,
                       justify_xlim=True,
                       legend_position='lower right',
                       legend_pad='not implemented',
                       print_value='auto',
                       legend_bbox_to_anchor=(0., 0.),
                       title=None):
        """
        Create a single barplot for each group of the first attribute in best.
        """
        no_subplots = len(best.index.levels[0])
        f, ax_arr = plt.subplots(no_rows,
                                 np.int(np.ceil(no_subplots * 1. / no_rows)))
        axes_flat = ax_arr.flatten()

        att_subplots, att_bars = best.index.names
        self._process_demanded_att_order()

        subplots = self._bring_in_order(best.index.levels[0], att_subplots)
        bars = self._bring_in_order(best.index.levels[1], att_bars)

        best = best.reset_index()
        if counts is not None:
            counts = counts.reset_index()

        cmap = plt.cm.get_cmap(cmap)
        legend_dummys = []

        color_cycling = self._color_cycl_f[att_bars] \
            if self._color_cycl_f.has_key(att_bars) else 1.


        for i_plt, (i_subplots, subplot_name) in enumerate(subplots):
            ax = axes_flat[i_plt]
            for bar_i, (i_bar, bar_name) in enumerate(bars):

                relevant_results = best[(best[att_subplots] == subplot_name)
                                        & (best[att_bars] == bar_name)]

                bar_color = cmap(
                    np.float((self._att_order[att_bars].index(bar_name))
                    * np.float(color_cycling)
                    / len(self._att_order[att_bars])) % 1.)

                legend_dummys.append(Rectangle((0, 0), 1, 1, fc=bar_color))

                if len(relevant_results) == 0:
                    continue

                # compute plot limits
                if plot_range:
                    bottom = plot_range[0]
                    ceil = plot_range[1]
                elif adapt_bottom:
                    rlvnt = best[(best[att_subplots] == subplot_name)
                                    & -(best['test mean'] == 0)]

                    if adapt_bottom == 'train':
                        data_max = np.max(rlvnt[['train mean', 'test mean']]).max()
                        data_min = np.min(rlvnt[['train mean', 'test mean']]).min()
                    else:
                        data_max = np.max(rlvnt['test mean'])
                        data_min = np.min(rlvnt['test mean'])

                    if error:
                        data_max += np.max(rlvnt['test std'])
                        data_min -= np.max(rlvnt['test std'])

                    ceil = misc.based_ceil(data_max + eps, base)
                    bottom = misc.based_floor(data_min - eps, base)

                test_mean = misc.float(relevant_results['test mean'])
                test_std = misc.float(relevant_results['test std'])
                train_mean = misc.float(relevant_results['train mean'])
                train_std = misc.float(relevant_results['train std'])

                bar_args = {'bottom': bottom,
                            'ecolor': 'gray',
                            'linewidth': 0.,
                            'color': bar_color,
                            'width': .4 if plot_fit else .8,
                            }
                text_args = {'ha': 'center',
                             'va': 'center',
                             'rotation': 'vertical'
                             }

                if error:
                    yerr_train = train_std
                    yerr_test = test_std
                else:
                    yerr_train = yerr_test = None

                if plot_fit:
                    ax.bar(bar_i, train_mean - bottom, yerr=yerr_train,
                           alpha=.5, **bar_args)
                    ax.bar(bar_i + .4, test_mean - bottom, yerr=yerr_test,
                           **bar_args)

                    if print_value is True \
                            or (print_value is not False and counts is None):
                        ax.text(bar_i + .25, (test_mean + bottom) / 2,
                                '%.2f' % train_mean,
                                **text_args)
                        ax.text(bar_i + .65, (test_mean + bottom) / 2,
                                '%.2f' % test_mean,
                                **text_args)

                else:
                    ax.bar(bar_i, test_mean - bottom, yerr=yerr_test,
                           **bar_args)

                    if print_value is True or \
                            (print_value is not False and counts is None):
                        ax.text(bar_i + .5, (test_mean + bottom) / 2,
                                '%.2f' % test_mean,
                                **text_args)


                # print count
                if counts is not None:
                    count = counts[(counts[att_subplots] == subplot_name)
                                   & (counts[att_bars] == bar_name)][
                        'test mean']

                    if count > 0:
                        ax.text(bar_i + .4, (test_mean + bottom) / 2,
                                '%d' % count,
                                **text_args)

                ax.set_title(subplot_name)
                ax.set_xticks([])
                if justify_xlim:
                    ax.set_xlim(0, len(bars))
                ax.set_ylim(bottom, ceil)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('gray')
                ax.spines['bottom'].set_color('gray')

        # if the number of axes is larger than the number of filled subplots,
        # remove superfluous axes
        for i_plt in range(len(subplots), len(axes_flat)):
            axes_flat[i_plt].axis('off')

        legend = [att for i, att in
                  bars]  # (int(att) if isinstance(att, float) else att)

        plt.figlegend(legend_dummys, legend, loc=legend_position,
                      ncol=legend_ncol, title=att_bars,
                      bbox_to_anchor=legend_bbox_to_anchor)

        if title:
            f.suptitle(title)

    def plot_hist_best_setting(self, best, figsize=[20, 10]):
        """
        Plots histograms of the parameter settings
        that lead to the best results.
        """
        parameters = pd.DataFrame()
        for value in best.values:
            ind = np.where(pd.np.all(self.results[best.columns] == value, 1))
            parameters = parameters.append(self.results.iloc[ind])

        if len(parameters) > 0:
            parameters.hist(
                column=list(set(parameters.columns) - set(best.columns)),
                figsize=figsize
            )
        else:
            print 'I cannot find any results that match. Did you use mean as' \
                  ' an objective?'

        return parameters
