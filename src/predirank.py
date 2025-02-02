"""
predirank

    Ranks the accuracy of classifiers on data set

"""

#   c4pub@git 2024
#
# Latest version available at: https://github.com/c4pub/
#

import datetime
import random
import sys
import traceback
import matplotlib.pyplot as plt

import math
import statistics
import numpy as np
import pandas as pd

# import deodel
# import deodel2
# import nngoa
# import usap_common
# import usap_csv_eval

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > C4pUseCommon - Begin
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def GetVersion() :
    return "v1.1.1"


class C4pUseCommon :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def flush() :
        """ Non delayed prints """

        import sys

        sys.stdout.flush()
        sys.stderr.flush()

        return True

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def iprnt( *args, **kwargs ) :
        """ Non delayed prints """

        ret_item = print( *args, **kwargs )
        C4pUseCommon.flush()

        return ret_item

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def SepLine1 ( display_flag = True ) :
        separator_string = ">-" + 78*"-"
        print ( separator_string )

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def SepLine2 ( display_flag = True ) :
        separator_string = 80*"-"
        print ( separator_string )

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def CrtTimeStamp(display_flag = True) :
        import datetime

        in_time_stamp = datetime.datetime.now()
        time_str = in_time_stamp.strftime("%Y-%m-%d %H:%M:%S")
        out_str = "time_stamp: %s" % (time_str)
        if display_flag :
            print(out_str)
        return out_str

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def ListToTabStr(in_data_list, in_tab_list = 8) :
        more_char = '>'
        space_char = ' '
        list_len = len(in_data_list)
        if not isinstance(in_tab_list, list) :
            use_tab_list = [in_tab_list] * (list_len)
        else :
            use_tab_list = in_tab_list
        total_str = ""
        for crt_idx in range(list_len - 1) :
            crt_elem = in_data_list[crt_idx]
            crt_tab = use_tab_list[crt_idx]
            data_width = crt_tab - 1
            crt_str = str(crt_elem)
            str_len = len(crt_str)
            if str_len == 0 :
                transf_str = (space_char)*(data_width)
            elif str_len > data_width :
                transf_str = crt_str[:(data_width - 1)] + more_char
            else :
                transf_str = crt_str + space_char * (data_width - str_len)
            total_str += (transf_str + space_char)
        # last column element can be any length
        transf_str = str(in_data_list[-1])
        total_str += (transf_str + space_char)
        return total_str

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def HashString(in_str = "") :
        total_sum = 0
        str_len = len(in_str)
        for crt_idx in range(str_len) :
            crt_elem = in_str[crt_idx]
            total_sum += (ord(crt_elem))
            total_sum ^= (crt_idx + 1)
        return total_sum

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def HashValue(in_param = None) :
        total_sum = 0
        param_str = str(in_param)
        param_str += str(type(in_param))
        hash_num = C4pUseCommon.HashString(param_str)
        return hash_num

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > C4pUseCommon - End
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > C4pSetMisc - Begin
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class C4pSetMisc :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def OrderedFreqCount(in_symbol_sequence_list):
        """
        Returns a list that contains info about the frequency of
        items in parameter input list (in_symbol_sequence_list).
        The list is sorted on no of occurences order

        Params:
            in_symbol_sequence_list
                sequence of symbol occurences

        returns:
            out_list
                The output list has the following structure:
                each row (first level list)  has:
                    first column the element itself from the list
                    second column the no of occurences
                    third column the list of indexes containing the element
        """
        from operator import itemgetter

        in_len = len(in_symbol_sequence_list)
        idx = 0
        out_list = []
        for in_el in in_symbol_sequence_list:
            found_match = 0
            for out_el in out_list:
                if in_el == out_el[0]:
                    out_el[1] = out_el[1] + 1
                    out_el[2].append(idx)
                    found_match = 1
                    break
            if found_match == 0:
                out_list.append([in_el, 1, []])
                # append index into third column
                out_list[-1][2].append(idx)
            idx = idx + 1

        out_list.sort(key=itemgetter(1), reverse=True)
        return out_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def SummaryFreqCount(in_symbol_sequence_list):
        """
        Returns a summary about the frequency of
        items in parameter input list (in_symbol_sequence_list).
        The list is sorted on no of occurences order

        Params:
            in_symbol_sequence_list
                sequence of symbol occurences

        returns:
            ret_no_of_distinct_elems
            ret_elem_list
                List of distinct elements
            ret_count_list
                List with counts of each element matching ret_elem_list
        """
        count_data = C4pSetMisc.OrderedFreqCount(in_symbol_sequence_list)
        distinct_elem_no, elem_list, count_list = C4pSetMisc.CountDataToFreqLists(count_data)

        return distinct_elem_no, elem_list, count_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def CountDataToFreqLists(in_freq_count_data):
        """
        Returns a summary of info about the frequency of
        items in parameter input list (in_freq_count_data).
        The lists are sorted on no of occurences order

        Params:
            in_freq_count_data
                sequence of symbol occurences

        returns:
            ret_no_of_distinct_elems
            ret_elem_list
                List of distinct elements
            ret_count_list
                List with counts of each element matching ret_elem_list
        """
        # determine no of distinct elements
        distinct_elem_no = len(in_freq_count_data)

        # Filter out the index lists
        elem_count_pairs = [ elem[:2] for elem in in_freq_count_data ]

        elem_list = [ x[0] for x in elem_count_pairs ]
        count_list = [ x[1] for x in elem_count_pairs ]

        return distinct_elem_no, elem_list, count_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > C4pSetMisc - End
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > C4pTblUtil - Begin
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class C4pTblUtil :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def GetCol( in_array, in_col ) :
        ret_item = []
        if not isinstance(in_array, list) :
            ret_item = None
        else :
            row_no = len(in_array)
            if not isinstance(in_array[0], list) :
                ret_item = None
            else :
                ret_item = []
                for crt_idx_row in range(row_no) :
                    crt_row_len = len(in_array[crt_idx_row])
                    if in_col < crt_row_len :
                        ret_item.append((in_array[crt_idx_row][in_col]))
                    else :
                        ret_item.append(None)
        return(ret_item)

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def MatrixTranspose( in_array ) :
        list_array = C4pTblUtil.ListDataConvert(in_array)
        if list_array == [] : return []
        if not isinstance(list_array[0], list) :
            transp_data = list(list_array)
        else :
            transp_data = []
            col_no = len(list_array[0])
            for crt_idx_col in range(col_no) :
                crt_vect = C4pTblUtil.GetCol( list_array, crt_idx_col )
                transp_data.append(crt_vect)
        ret_item = transp_data
        return(ret_item)

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def NumericalCheck( in_value, int_is_num_flag = True ) :
        """
            Check if numerical. If non regular float, result is false and
            translated value returned
        """
        if not int_is_num_flag :
            if isinstance(in_value, float) :
                float_flag, valid_val = C4pTblUtil.ValidateFloat(in_value)
                fn_ret = (float_flag, valid_val)
            else :
                fn_ret = (False, in_value)
        else :
            if isinstance(in_value, float) :
                float_flag, valid_val = C4pTblUtil.ValidateFloat(in_value)
                fn_ret = (float_flag, valid_val)
            elif isinstance(in_value, int) :
                fn_ret = (True, in_value)
            else :
                fn_ret = (False, in_value)
        return fn_ret

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def ValidateFloat(in_val) :
        if not isinstance(in_val, float) :
            fn_ret_status = False
            fn_ret_translate = in_val
        elif np.isnan(in_val) :
            fn_ret_status = False
            fn_ret_translate = "nan"
        elif in_val == float('inf') :
            fn_ret_status = False
            fn_ret_translate = "+inf"
        elif in_val == float('-inf') :
            fn_ret_status = False
            fn_ret_translate = "-inf"
        else :
            fn_ret_status = True
            fn_ret_translate = in_val
        ret_tuple = fn_ret_status, fn_ret_translate
        return ret_tuple

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def InternalListConvert(in_data) :
        if (isinstance(in_data, list)) :
            lst_data = in_data.copy()
        elif (isinstance(in_data, np.ndarray)) :
            lst_data = in_data.tolist()
        elif (isinstance(in_data, pd.core.arrays.PandasArray)) :
            lst_data = in_data.tolist()
        elif (isinstance(in_data, pd.core.frame.DataFrame)) :
            lst_data = in_data.values.tolist()
        elif (isinstance(in_data, pd.core.series.Series)) :
            lst_data = in_data.values.tolist()
        else :
            lst_data = in_data

        return lst_data

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def ListDataConvert(in_data) :
        if (isinstance(in_data, list)) :
            # check whether rows are also lists
            len_data = len(in_data)
            if len_data > 0 :
                first_row = in_data[0]
                if (isinstance(first_row, list)) :
                    lst_data = in_data.copy()
                else :
                    lst_data = []
                    for crt_row in in_data :
                        new_row = C4pTblUtil.InternalListConvert(crt_row)
                        lst_data.append(new_row)
            else :
                lst_data = []
        else :
            lst_data = C4pTblUtil.InternalListConvert(in_data)
        return lst_data

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def StrToNum(in_str) :
        # string to number preserving int or float representation type
        # import deodel

        if not isinstance(in_str, str) :
            return None
        try :
            float_number = float(in_str)
            valid_flag, translated_val = C4pTblUtil.ValidateFloat(float_number)
            if not valid_flag :
                return None
            int_number = int(float_number)
            if int_number == float_number :
                # potential integer
                if ('.' in in_str) or ('e' in in_str) or ('E' in in_str) :
                    ret_number = float_number
                else :
                    # preserve integer type
                    ret_number = int_number
            else :
                ret_number = float_number
            return ret_number
        except ValueError:
            return None

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def StrToSubstr(in_str) :
        import re

        if not isinstance(in_str, str) :
            return None
        substr_flag = False
        sub_str = None
        cpy_instr = in_str
        trim_str = cpy_instr.strip()
        ret_reg = re.search(r"^[\"](?P<tmptag>.*)[\"]$", trim_str)
        if ret_reg != None :
            substr_flag = True
            sub_str = ret_reg.group('tmptag')
        else :
            ret_reg = re.search(r"^[\'](?P<tmptag>.*)[\']$", trim_str)
            if ret_reg != None :
                substr_flag = True
                sub_str = ret_reg.group('tmptag')
        return sub_str

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def CsvTblPrep(in_raw_tbl) :
        # Prepare csv table validating cell content
        # import deodel

        row_no = len(in_raw_tbl)
        tbl_csv = []
        # convert fields
        for crt_row_idx in range(row_no) :
            row_crt = in_raw_tbl[crt_row_idx]
            row_new = []
            if not isinstance(row_crt, list) :
                if not row_crt == None :
                    row_new.append(str(row_crt))
                else :
                    row_new.append(None)
            else :
                col_no = len(row_crt)
                for crt_col_idx in range(col_no) :
                    crt_elem = in_raw_tbl[crt_row_idx][crt_col_idx]
                    if isinstance(crt_elem, int) :
                        new_elem = crt_elem
                    elif isinstance(crt_elem, float) :
                        float_flag, translated_val = C4pTblUtil.ValidateFloat(crt_elem)
                        new_elem = translated_val
                    elif not isinstance(crt_elem, str) :
                        if not crt_elem == None :
                            new_elem = str(crt_elem)
                        else :
                            new_elem = None
                    elif crt_elem == '' :
                        new_elem = None
                    else :
                        num_elem = C4pTblUtil.StrToNum(crt_elem)
                        if num_elem == None :
                            substr_elem = C4pTblUtil.StrToSubstr(crt_elem)
                            if substr_elem == None :
                                new_elem = crt_elem
                            else :
                                new_elem = substr_elem
                        else :
                            new_elem = num_elem
                    row_new.append(new_elem)
            tbl_csv.append(row_new)
        return tbl_csv

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def GetFileCsv(in_csv_local) :
        import csv

        fn_ret_status = False
        fn_ret_data = None
        fn_ret_msg = ""

        # one iteration loop to allow unified return through loop breaks
        for dummy_idx in range(1) :
            try :
                file_d = open(in_csv_local, 'r')
                csv_read = csv.reader(file_d)
            except IOError :
                fn_ret_msg = "Error: failed file access !"
                break
            else :
                fn_ret_status = True
                fn_ret_data = csv_read

        fn_ret_tuple = (fn_ret_status, fn_ret_data, fn_ret_msg)
        return fn_ret_tuple

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def ImportCsvTbl(in_csv_path) :
        fn_ret_status = False
        fn_ret_data = None
        fn_ret_msg = ""

        # one iteration loop to allow unified return through loop breaks
        for dummy_idx in range(1) :
            csv_read = None
            ret_info = C4pTblUtil.GetFileCsv(in_csv_path)
            ret_status, ret_data, ret_msg = ret_info
            if ret_status :
                csv_read = ret_data
            else :
                fn_ret_msg = ret_msg
            if csv_read == None :
                pass
            else :
                list_csv = list(csv_read)
                fn_ret_status = True
                fn_ret_data = list_csv

        fn_ret_tuple = (fn_ret_status, fn_ret_data, fn_ret_msg)
        return fn_ret_tuple

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def TblPrepProxy(in_csv) :
        ret_status = False
        tbl_csv = None
        ret_msg = ""
        if isinstance(in_csv, str) :
            # input is a url or file path
            ret_status, list_csv, ret_msg = C4pTblUtil.ImportCsvTbl(in_csv)
            if not ret_status :
                return ret_status, tbl_csv, ret_msg
        else :
            # assumes input is a list of lists table
            list_csv = in_csv[:]
            ret_status = True
        if ret_status :
            tbl_csv = C4pTblUtil.CsvTblPrep(list_csv)
        else :
            ret_msg = "Error: csv data could not be accessed !"
        return ret_status, tbl_csv, ret_msg

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def UniformCleanRows(in_raw_tbl, in_exclude_set) :
        # Weed out empty rows and patch missing columns cells.

        fn_ret_dict = {'status': False, 'proc_tbl': [], 'del_row': [], 'dim':(0, 0)}
        clean_row_tbl = []
        if in_raw_tbl == [] :
            return fn_ret_dict
        row_no = len(in_raw_tbl)
        col_max = len(in_raw_tbl[0])
        col_min = len(in_raw_tbl[0])
        initial_row_no = row_no
        # search the maximum column no and remove empty lines
        for crt_idx in range(row_no) :
            crt_row = in_raw_tbl[crt_idx]
            col_no = len(crt_row)
            if col_no > col_max :
                col_max = col_no
            if col_no < col_min :
                col_min = col_no
            if crt_idx in in_exclude_set :
                # remove request
                remove_flag = True
            elif crt_row == [None]*col_no :
                # skip empty row
                remove_flag = True
            else :
                remove_flag = False
            if remove_flag :
                fn_ret_dict['del_row'].append(crt_idx)
            else :
                clean_row_tbl.append(crt_row)
        if col_min != col_max :
            # reparse to adjust rows
            row_no = len(clean_row_tbl)
            clean_tbl = []
            for crt_idx in range(row_no) :
                crt_row = clean_row_tbl[crt_idx]
                col_no = len(crt_row)
                new_row = crt_row[:]
                delta_col = col_max - col_no
                if delta_col > 0 :
                    # append columns
                    append_list = [None]*(delta_col)
                    new_row = crt_row + append_list
                clean_tbl.append(new_row)
        else :
            clean_tbl = clean_row_tbl
        fn_ret_dict['status'] = True
        fn_ret_dict['proc_tbl'] = clean_tbl
        fn_ret_dict['dim'] = (initial_row_no, col_max)
        return fn_ret_dict

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def AccuracyEval(x_data, y_target, classifier, iterations = 1, train_fraction = 0.5, random_seed = 42, aux_data = None, display_flag = True) :

        import random
        data_rows = len(x_data)
        data_cols = len(x_data[0])

        if display_flag : print("- - - - prediction accuracy test")
        if display_flag : print()
        if display_flag : print("- - - - - - classifier:", classifier)
        if display_flag : print("- - - - - - iterations:", iterations)
        if display_flag : print("- - - - - - train_fraction:", train_fraction)
        if display_flag : print("- - - - - - aux_data:", aux_data)
        if display_flag : print("- - - - - - random_seed:", random_seed)
        if display_flag : print()

        if aux_data == None :
            aux_data = {}

        random.seed(random_seed)

        ret_tuple = C4pTblUtil.AccuracyIterEval(x_data, y_target, classifier, iterations, train_fraction, random_seed, aux_data, True)
        avg_accuracy, rnd_accuracy, delta_secs, sample_test, sample_pred = ret_tuple

        column_limit = 40
        str_limit = 60
        if avg_accuracy == None :
            str_smpl_test = str(sample_test)
            str_smpl_pred = str(sample_pred)
        else :
            str_smpl_test = str(list(sample_test[:column_limit]))[:str_limit]
            str_smpl_pred = str(list(sample_pred[:column_limit]))[:str_limit]

        if display_flag : print("- - - - - - delta_secs:", delta_secs)
        if display_flag : print("- - - - - - sample_test:", str_smpl_test, "...")
        if display_flag : print("- - - - - - sample_pred:", str_smpl_pred, "...")
        if display_flag : print()
        if display_flag : print("- - - - - - avg_accuracy:", avg_accuracy)
        if display_flag : print("- - - - - - rnd_accuracy:", rnd_accuracy)
        return avg_accuracy, rnd_accuracy

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def AccuracyIterEval(x_data, y_target, classifier, iterations, train_fraction, random_seed, aux_data, valid_target_flag) :
        import datetime
        import random
        from sklearn.model_selection import train_test_split
        fn_ret_tuple = None, None, 0, None, None

        crt_time_ref = datetime.datetime.now()
        test_fraction = 1.0 - train_fraction
        cumulate_clsf_accuracy = 0
        cumulate_rand_accuracy = 0
        crt_rand_seed = random_seed
        if len(x_data) <= 1 :
            return fn_ret_tuple
        if not len(x_data) == len(y_target) :
            return fn_ret_tuple

        for crt_idx in range(iterations) :
            if not random_seed == None :
                crt_rand_seed = random_seed + crt_idx
            fn_ret_tuple = train_test_split(x_data, y_target, test_size = test_fraction, random_state = crt_rand_seed)
            x_train, x_test, y_train, y_test = fn_ret_tuple
            classifier.fit(x_train, y_train)
            y_predict = classifier.predict(x_test)

            # compute accuracy
            test_total_no = len(y_test)
            train_total_no = len(y_train)
            total_clsf_matches = 0
            total_rand_matches = 0
            effective_total = test_total_no
            rand_list = random.sample(range(0, train_total_no), train_total_no)
            for crt_jdx in range(test_total_no) :
                if valid_target_flag and y_test[crt_jdx] == None :
                    effective_total -= 1
                    continue
                if y_test[crt_jdx] == y_predict[crt_jdx] :
                    total_clsf_matches += 1
                rand_idx = rand_list[crt_jdx % train_total_no]
                if y_test[crt_jdx] == y_train[rand_idx] :
                    total_rand_matches += 1

            if effective_total > 0 :
                clsf_accuracy = total_clsf_matches / (1.0 * effective_total)
                cumulate_clsf_accuracy += clsf_accuracy
                rand_accuracy = total_rand_matches / (1.0 * effective_total)
                cumulate_rand_accuracy += rand_accuracy

            if crt_idx == 0 :
                sample_test = y_test[:]
                sample_pred = y_predict[:]

        clsf_avg_accuracy = (cumulate_clsf_accuracy * 1.0) / iterations
        rand_avg_accuracy = (cumulate_rand_accuracy * 1.0) / iterations

        new_time_ref = datetime.datetime.now()
        delta = new_time_ref - crt_time_ref
        delta_secs = delta.total_seconds()

        fn_ret_tuple = clsf_avg_accuracy, rand_avg_accuracy, delta_secs, sample_test, sample_pred
        return fn_ret_tuple

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def CleanTargetExtract(in_table, in_targ_idx = None, in_exclude_col_set = None) :
        # import deodel

        fn_ret_status = False
        fn_ret_table = []
        fn_ret_col = []
        fn_ret_dim = (0, 0)
        fn_ret_msg = "Error: data extraction failed !"

        crt_tbl = in_table
        # Weed out empty rows and patch missing columns cells.
        ret_info = C4pTblUtil.UniformCleanRows(crt_tbl, {})
        if not ret_info['status'] :
            fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg
            return fn_ret_tuple
        crt_tbl = ret_info['proc_tbl']
        tbl_dim = ret_info['dim']
        fn_ret_dim = tbl_dim

        # set target col idx
        if in_targ_idx == None :
            # in_targ_idx = -1
            last_column = tbl_dim[1] - 1
            crt_targ_col = last_column
        else :
            crt_targ_col = in_targ_idx

        if in_exclude_col_set == None :
            in_exclude_col_set = {}

        # Weed out empty columns.
        transpose_tbl = C4pTblUtil.MatrixTranspose(crt_tbl)
        ret_info = C4pTblUtil.UniformCleanRows(transpose_tbl, in_exclude_col_set)
        trans_crt_tbl = ret_info['proc_tbl']

        # adjust target column if required
        adj_list = ret_info['del_row']
        adj_len = len(adj_list)
        adj_idx = 0
        exit_flag = False
        for crt_idx in range(adj_len) :
            old_idx = adj_list[crt_idx]
            if old_idx > crt_targ_col :
                # changes past target column are not relevant
                break
            elif old_idx == crt_targ_col :
                # the chosen target column was a removed empty list
                fn_ret_msg = "Error: empty target column !"
                exit_flag = True
                break
            else :
               adj_idx += 1
        if exit_flag :
            fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg
            return fn_ret_tuple
        new_targ_col = crt_targ_col - adj_idx

        if new_targ_col >= len(trans_crt_tbl) :
            fn_ret_msg = "Error: invalid target index !"
            fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg
            return fn_ret_tuple

        # split table into attributes and target
        target_col = trans_crt_tbl.pop(new_targ_col)
        trans_train_attr = trans_crt_tbl

        # restore train transposed matrix
        train_tbl = C4pTblUtil.MatrixTranspose(trans_train_attr)

        fn_ret_status = True
        fn_ret_table = train_tbl
        fn_ret_col = target_col
        fn_ret_dim = tbl_dim
        fn_ret_msg = ""
        fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg

        return fn_ret_tuple

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > C4pTblUtil - End
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > RandPredictor - Begin
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class RandPredictor:
    """Random baseline predictor.

    Randomly chooses a prediction from the training outputs.

    Parameters
    ----------
    aux_param : dict, default=None
        Auxiliary configuration parameters.
        Configuration dictionary keywords:
            'rand_seed' : int, default=None
                Random seed. If "None" no seed initialization

    Attributes
    ----------
    version : float
        Version of algorithm implementation

    Notes
    -----
    Can be used for both classification or regression.

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0.5, 0, 'a', 4.2]
    >>> from randbaseline import RandPredictor:

    >>> randbaseline = RandPredictor:
()
    >>> randbaseline.fit(X, y)
    RandPredictor:
(...)
    >>> print(randbaseline.predict([[1.1]]))
    [4.2]
    """

    def __init__(
        self,
        aux_param = None
    ):
        if aux_param == None :
            self.aux_param = {}
        else :
            self.aux_param = aux_param

    version = 0.01

    def __repr__(self):
        '''Returns representation of the object'''
        return("{}({!r})".format(self.__class__.__name__, self.aux_param))

    def fit(self, X, y):
        """Fit the classifier with the training dataset.

        Parameters
        ----------
        X : array-like matrix of shape (n_samples, n_features)
            This parameter is ignored, present only for compatibility.

        y : array-like matrix of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : RandPredictor:

            The fitted classifier.
        """

        ret = RandPredictor.WorkFit( self, X, y )
        return ret

    def predict(self, X):
        """Random prediction.

        Parameters
        ----------
        X : array-like of shape n_queries, n_features
            Test samples.

        Returns
        -------
        y : list of (n_queries,)
            Class labels for each data sample.
        """

        ret = RandPredictor.WorkPredict( self, X )
        return ret

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def WorkFit(object, in_X, in_y):
        import random

        data_y = in_y.copy()
        aux_param = object.aux_param

        r_state = random.getstate()
        if 'rand_seed' in aux_param :
            rand_seed = aux_param['rand_seed']
            random.seed(rand_seed)
        train_total_no = len(data_y)
        rand_list = random.sample(range(0, train_total_no), train_total_no)
        random.setstate(r_state)

        object.rand_idx_list = rand_list
        object.train_y = data_y

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def WorkPredict(object, in_query):

        query_req = in_query[:]

        train_total_no = len(object.train_y)
        query_len = len(query_req)
        rand_list = object.rand_idx_list
        y_out_lst = [None]*query_len

        for crt_idx in range(query_len) :
            hash_query = C4pUseCommon.HashValue(query_req[crt_idx])
            rand_idx = rand_list[(hash_query + crt_idx) % train_total_no]
            y_out_lst[crt_idx] = object.train_y[rand_idx]

        return y_out_lst

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# > RandPredictor - End
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# from usap_common import *

iprnt = C4pUseCommon.iprnt

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def RowListDisplay(in_row_list) :

    print("- - - - ")
    for row in in_row_list :
        print("- - - - - - - - row:", row)
    print("- - - - ")
    print()

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def FillMissing(defaults, values) :
    values_copy = values[:]
    for i in range(len(defaults)) :
        if i >= len(values_copy) :
            values_copy.append(defaults[i])
    return values_copy

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def IsRowSimilarCompare(in_head_row, in_first_row) :

    fn_ret_data = None

    # one iteration loop to allow unified return through loop breaks
    for dummy_idx in range(1) :

        row_head_len = len(in_head_row)
        row_first_len = len(in_first_row)
        if not row_head_len == row_first_len :
            fn_ret_data = None
            break
        similar_flag = True
        for crt_idx in range(row_head_len) :
            crt_elem_head = in_head_row[crt_idx]
            crt_elem_first = in_first_row[crt_idx]
            if crt_elem_head == None or crt_elem_first == None :
                continue
            else :
                if not type(crt_elem_head) == type(crt_elem_first) :
                    is_num_head, translate_value = C4pTblUtil.NumericalCheck(crt_elem_head, int_is_num_flag = True)
                    is_num_first, translate_value = C4pTblUtil.NumericalCheck(crt_elem_first, int_is_num_flag = True)
                    if is_num_head and is_num_first :
                        # both are numbers
                        continue
                    else :
                        similar_flag = False
                        break
        fn_ret_data = similar_flag
    return fn_ret_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def UniformCleanRows(in_raw_tbl, in_exclude_set) :
    # Weed out empty rows and patch missing columns cells.

    fn_ret_dict = {'status': False, 'proc_tbl': [], 'del_row': [], 'dim':(0, 0)}
    clean_row_tbl = []
    if in_raw_tbl == [] :
        return fn_ret_dict
    row_no = len(in_raw_tbl)
    col_max = len(in_raw_tbl[0])
    col_min = len(in_raw_tbl[0])
    initial_row_no = row_no
    # search the maximum column no and remove empty lines
    for crt_idx in range(row_no) :
        crt_row = in_raw_tbl[crt_idx]
        col_no = len(crt_row)
        if col_no > col_max :
            col_max = col_no
        if col_no < col_min :
            col_min = col_no
        if crt_idx in in_exclude_set :
            # remove request
            remove_flag = True
        elif crt_row == [None]*col_no :
            # skip empty row
            remove_flag = True
        else :
            remove_flag = False
        if remove_flag :
            fn_ret_dict['del_row'].append(crt_idx)
        else :
            clean_row_tbl.append(crt_row)
    if col_min != col_max :
        # reparse to adjust rows
        row_no = len(clean_row_tbl)
        clean_tbl = []
        for crt_idx in range(row_no) :
            crt_row = clean_row_tbl[crt_idx]
            col_no = len(crt_row)
            new_row = crt_row[:]
            delta_col = col_max - col_no
            if delta_col > 0 :
                # append columns
                append_list = [None]*(delta_col)
                new_row = crt_row + append_list
            clean_tbl.append(new_row)
    else :
        clean_tbl = clean_row_tbl
    fn_ret_dict['status'] = True
    fn_ret_dict['proc_tbl'] = clean_tbl
    fn_ret_dict['dim'] = (initial_row_no, col_max)
    return fn_ret_dict

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def CleanTargetExtract(in_table, in_targ_idx = None, in_exclude_col_set = None) :
    # import deodel

    fn_ret_status = False
    fn_ret_table = []
    fn_ret_col = []
    fn_ret_dim = (0, 0)
    fn_ret_msg = "Error: data extraction failed !"

    crt_tbl = in_table
    # Weed out empty rows and patch missing columns cells.
    ret_info = UniformCleanRows(crt_tbl, {})
    if not ret_info['status'] :
        fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg
        return fn_ret_tuple
    crt_tbl = ret_info['proc_tbl']
    tbl_dim = ret_info['dim']
    fn_ret_dim = tbl_dim

    # set target col idx
    if in_targ_idx == None :
        # in_targ_idx = -1
        last_column = tbl_dim[1] - 1
        crt_targ_col = last_column
    else :
        crt_targ_col = in_targ_idx

    if in_exclude_col_set == None :
        in_exclude_col_set = {}

    # Weed out empty columns.
    transpose_tbl = C4pTblUtil.MatrixTranspose(crt_tbl)
    ret_info = UniformCleanRows(transpose_tbl, in_exclude_col_set)
    trans_crt_tbl = ret_info['proc_tbl']

    # adjust target column if required
    adj_list = ret_info['del_row']
    adj_len = len(adj_list)
    adj_idx = 0
    exit_flag = False
    for crt_idx in range(adj_len) :
        old_idx = adj_list[crt_idx]
        if old_idx > crt_targ_col :
            # changes past target column are not relevant
            break
        elif old_idx == crt_targ_col :
            # the chosen target column was a removed empty list
            fn_ret_msg = "Error: empty target column !"
            exit_flag = True
            break
        else :
           adj_idx += 1
    if exit_flag :
        fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg
        return fn_ret_tuple
    new_targ_col = crt_targ_col - adj_idx

    if new_targ_col >= len(trans_crt_tbl) :
        fn_ret_msg = "Error: invalid target index !"
        fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg
        return fn_ret_tuple

    # split table into attributes and target
    target_col = trans_crt_tbl.pop(new_targ_col)
    trans_train_attr = trans_crt_tbl

    # restore train transposed matrix
    train_tbl = C4pTblUtil.MatrixTranspose(trans_train_attr)

    fn_ret_status = True
    fn_ret_table = train_tbl
    fn_ret_col = target_col
    fn_ret_dim = tbl_dim
    fn_ret_msg = ""
    fn_ret_tuple = fn_ret_status, fn_ret_table, fn_ret_col, fn_ret_dim, fn_ret_msg

    return fn_ret_tuple

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def CvsExtractData(in_csv_data, in_targ_idx = -1, aux_data = None, display_flag = False) :
    # import deodel

    fn_ret_data = None, None

    # one iteration loop to allow unified return through loop breaks
    for dummy_idx in range(1) :

        list_csv = in_csv_data

        # Provide default values where needed
        if in_targ_idx == None :
            in_targ_idx = -1
        if aux_data == None :
            aux_data = {}
        if display_flag == None :
            display_flag = True

        # show data
        str_max_len = 60
        row_max_len = 6
        tbl_sample = list_csv[:row_max_len]
        if display_flag: print()
        if display_flag: print("- - - - list_csv:")
        for crt_row in tbl_sample :
            str_row = str(crt_row)[:str_max_len] + " ..."
            if display_flag: print("- - - - - - - - crt_row:", str_row)
        if display_flag: print("                ...")

        if 'exc' in aux_data :
            exclude_column_set = aux_data['exc']
        else :
            exclude_column_set = {}

        ret_info = CleanTargetExtract(list_csv, in_targ_idx, exclude_column_set)
        ret_status, train_tbl, target_col, ret_dim, ret_str = ret_info

        tbl_rows = ret_dim[0]
        tbl_cols = ret_dim[1]
        if display_flag: print("- - - - - - - - tbl_rows:", tbl_rows)
        if display_flag: print("- - - - - - - - tbl_cols:", tbl_cols)

        if not ret_status :
            break

        row_max_len = 4
        tbl_sample = train_tbl[:row_max_len]
        if display_flag: print()
        if display_flag: print("- - - - train_tbl:")
        for crt_row in tbl_sample :
            str_row = str(crt_row)[:str_max_len] + " ..."
            if display_flag: print("- - - - - - - - crt_row:", str_row)

        str_row = str(target_col)[:str_max_len] + " ..."
        if display_flag: print()
        if display_flag: print("- - - - target_col:", str_row)
        if display_flag: print()

        # ret_data = (data_digi_x, data_target_y)
        fn_ret_data = train_tbl, target_col

    return fn_ret_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def SelectiveOneHotProc(in_vector, in_top_no) :

    fn_ret_data = []
    # one iteration loop to allow unified return through loop breaks
    for dummy_idx in range(1) :

        vect_len = len(in_vector)
        ret_tuple = C4pSetMisc.SummaryFreqCount(in_vector)
        crt_types_no, crt_id_list, crt_count_list = ret_tuple
        label_no = min(crt_types_no - 1, in_top_no)
        if label_no == 0 :
            one_hot_list = [[0] * vect_len]
        else :
            one_hot_list = []
            for crt_label_idx in range(label_no) :
                one_hot_list.append([])
            for crt_idx in range(vect_len) :
                crt_elem = in_vector[crt_idx]
                for crt_label_idx in range(label_no) :
                    crt_label_id = crt_id_list[crt_label_idx]
                    if crt_elem == crt_label_id :
                        append_elem = 1
                    else :
                        append_elem = 0
                    one_hot_list[crt_label_idx].append(append_elem)
        fn_ret_data = one_hot_list
    return fn_ret_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def AnalyzeVectorType(in_vector) :

    fn_ret_data = (False, False, [])
    # one iteration loop to allow unified return through loop breaks
    for dummy_idx in range(1) :

        opmode_intisnum = True
        has_num_flag = False
        has_categ_flag = False
        has_none_flag = False
        vect_len = len(in_vector)
        num_list = []
        for crt_idx in range(vect_len) :
            crt_elem = in_vector[crt_idx]
            is_numerical, translate_value = C4pTblUtil.NumericalCheck(crt_elem, opmode_intisnum)
            if is_numerical :
                num_list.append(crt_elem)
                has_num_flag = True
            elif crt_elem == None :
                has_none_flag = True
            else :
                has_categ_flag = True
        is_numerical = not has_categ_flag
        if is_numerical :
            if has_none_flag :
                list_avg = statistics.mean(num_list)
                convert_list = []
                for crt_idx in range(vect_len) :
                    crt_elem = in_vector[crt_idx]
                    if crt_elem == None :
                        convert_list.append(list_avg)
                    else :
                        convert_list.append(crt_elem)
                out_list = convert_list
            else :
                out_list = in_vector[:]
        else :
            # out_list = [None*crt_elem]
            out_list = in_vector[:]
        fn_ret_data = is_numerical, has_none_flag, out_list
    return fn_ret_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def ExtractColFromTbl(in_list_tbl, in_extract_idx) :

    transpose_tbl = C4pTblUtil.MatrixTranspose(in_list_tbl)

    # split table into target and remaining
    extract_col = transpose_tbl.pop(in_extract_idx)
    remain_list = transpose_tbl[:]

    # restore transposed matrix
    remain_tbl = C4pTblUtil.MatrixTranspose(remain_list)
    return extract_col, remain_tbl

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def RegularizeListTbl(in_list_tbl) :

    max_col_no = 0
    for crt_row in in_list_tbl :
        if isinstance(crt_row, list) :
            crt_len = len(crt_row)
            if crt_len > max_col_no :
                max_col_no = crt_len
    regular_tbl = []
    for crt_row in in_list_tbl :
        if isinstance(crt_row, list) :
            crt_len = len(crt_row)
        else :
            crt_len = 0
            crt_row = []
        if crt_len < max_col_no :
            delta_col = max_col_no - crt_len
            none_list = []
            for crt_idx in range(delta_col) :
                none_list.append(None)
            new_row = crt_row[:] + none_list
        else :
            new_row = crt_row[:]
        regular_tbl.append(new_row)
    return regular_tbl

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def GetConvertMixToNumTbl(in_list_tbl, in_onehot_col) :

    # print("GetConvertMixToNumTbl")
    row_no = len(in_list_tbl)
    col_no = len(in_list_tbl[0])

    transp_tbl = C4pTblUtil.MatrixTranspose(in_list_tbl)
    new_tbl = []
    insert_list = []
    for crt_idx in range(col_no) :
        crt_col = transp_tbl[crt_idx]
        ret_data = AnalyzeVectorType(crt_col)
        is_numerical, has_none_flag, out_list = ret_data
        if not is_numerical :
            one_hot_lst = SelectiveOneHotProc(out_list, in_onehot_col)
            insert_no = len(one_hot_lst)
            new_tbl = new_tbl + one_hot_lst
            insert_list.append((crt_idx, insert_no))
        else :
            new_tbl.append(out_list)
    ret_tbl = C4pTblUtil.MatrixTranspose(new_tbl)
    return ret_tbl, insert_list

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def RetrieveCsvTbl(in_csv, in_head_first = None) :

    # input is a url or file path
    ret_status, list_csv, ret_msg = C4pTblUtil.ImportCsvTbl(in_csv)

    tbl_csv = RegularizeListTbl(list_csv)
    list_reg_csv = C4pTblUtil.CsvTblPrep(tbl_csv)

    if in_head_first == None :
        sel_head_row = list_reg_csv[0]
        sel_next_row = list_reg_csv[1]
        is_similar = IsRowSimilarCompare(sel_head_row, sel_next_row)
        if is_similar :
            head_start_row = 0
        else :
            head_start_row = 1
    else :
        head_start_row = in_head_first
    list_csv = list_reg_csv[head_start_row:]
    return list_csv, head_start_row

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def TblTargetExtract(in_csv_tbl, in_targ_idx = None) :

    if in_targ_idx == None :
        in_targ_idx = -1
    list_csv = in_csv_tbl[:]

    tbl_csv = RegularizeListTbl(list_csv)
    list_csv = C4pTblUtil.CsvTblPrep(tbl_csv)
    targ_col, attr_tbl = ExtractColFromTbl(list_csv, in_targ_idx)
    return targ_col, attr_tbl

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def TransformTblOneHotEncoding(in_data_tbl, in_max_onehot = None) :

    if in_max_onehot == None :
        in_max_onehot = 4

    tbl_data = C4pTblUtil.ListDataConvert(in_data_tbl)
    onehot_tbl, insert_list = GetConvertMixToNumTbl(tbl_data, in_max_onehot)
    return onehot_tbl, insert_list

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def TransformTblDataSizeLimit(in_data_attr, in_data_targ, in_data_limit_row_max = 0,
                                in_data_limit_row_min = 0, in_data_limit_col_max = 0,
                                in_rand_seed = 42 ) :

    crt_data_attr = in_data_attr[:]
    crt_data_targ = in_data_targ[:]

    attr_no_row = len(crt_data_attr)
    attr_no_col = len(crt_data_attr[0])

    new_attr_data = crt_data_attr
    new_y_data = crt_data_targ

    if in_data_limit_row_min > 0 :
        if attr_no_row < in_data_limit_row_min :
            # dataset is too small, multiply it
            augment_factor = int((in_data_limit_row_min * 1.0) / attr_no_row) + 1
            augment_data_attr = crt_data_attr * augment_factor
            augment_targ_attr = crt_data_targ * augment_factor
            crt_data_attr = augment_data_attr[:]
            crt_data_targ = augment_targ_attr[:]
            attr_no_row *= augment_factor
            new_attr_data = crt_data_attr
            new_y_data = crt_data_targ

    if in_data_limit_row_max > 0 :
        if attr_no_row > in_data_limit_row_min :
            if attr_no_row > in_data_limit_row_max :
                # need to reduce the data set
                # in_rand_seed = 42
                reduction_ratio = in_data_limit_row_max / (1.0 * attr_no_row)
                reduced_row_no = int(reduction_ratio * attr_no_row)
                if ( reduced_row_no > in_data_limit_row_min
                     and reduced_row_no < attr_no_row ) :
                    new_attr_data = RandomShrinkTbl( crt_data_attr, reduced_row_no, in_rand_seed )
                    new_y_data = RandomShrinkTbl( crt_data_targ, reduced_row_no, in_rand_seed )
                    crt_data_attr = new_attr_data
                    crt_data_targ = new_y_data

    if in_data_limit_col_max > 0 :
        if attr_no_col > 0 :
            if attr_no_col > in_data_limit_col_max :
                # need to reduce the data set
                # in_rand_seed = 42
                reduction_ratio = in_data_limit_col_max / (1.0 * attr_no_col)
                reduced_col_no = int(reduction_ratio * attr_no_col)
                if ( reduced_col_no > 0
                     and reduced_col_no < attr_no_col ) :
                    transp_attr_data = C4pTblUtil.MatrixTranspose(crt_data_attr)
                    shrink_attr_data = RandomShrinkTbl( transp_attr_data, reduced_col_no, in_rand_seed )
                    new_attr_data = C4pTblUtil.MatrixTranspose(shrink_attr_data)

    return new_attr_data, new_y_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# @staticmethod
def GetQcutBin( in_num_v, in_bin_no ) :
    """
        Return cut bins for the numerical input vector
    """
    import pandas as pd
    # import numpy as np

    pd_qcut = pd.qcut(in_num_v, in_bin_no, retbins=True, labels=False, duplicates='drop')
    return(pd_qcut)

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# @staticmethod
def BinDiscretize( in_num_v, in_qcut_bins ) :
    """
        Discretize the numeric input vector according with bin thresholds
    """
    # import pandas as pd
    # import numpy as np

    qcut_len = len( in_qcut_bins )
    if qcut_len == 1 :
        num_len = len(in_num_v)
        ordinal_v = np.array(num_len * [0])

    elif qcut_len == 2 :
        adj_bins = in_qcut_bins
        adj_bins[0] = -np.inf
        adj_bins = np.concatenate((adj_bins, [np.inf]))
        ordinal_v = pd.cut( in_num_v, bins=adj_bins, labels=False, include_lowest=True, duplicates='drop')
    else :
        adj_bins = in_qcut_bins
        adj_bins[0] = -np.inf
        adj_bins[-1] = np.inf
        ordinal_v = pd.cut( in_num_v, bins=adj_bins, labels=False, include_lowest=True, duplicates='drop')
    return(ordinal_v)

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# @staticmethod
def DiscretizeVect( in_num_v, in_bin_no ) :
    """
        Return bin discretized numerical vector
    """
    # import pandas as pd
    # import numpy as np

    pd_qcut = GetQcutBin(in_num_v, in_bin_no)
    discretized_v = BinDiscretize(in_num_v, pd_qcut[1])
    return(discretized_v)

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# @staticmethod
def TransformTblDiscretize( in_tbl_data, in_bin_no ) :
    """
        Return bin discretized numerical vector
    """

    if ( in_bin_no == None or in_bin_no == 0 ) :
        return in_tbl_data

    num_arr_data = np.array(in_tbl_data)

    row_no, col_no = num_arr_data.shape
    data_digi_x = np.zeros((row_no, col_no))
    for crt_idx in range(col_no) :
        crt_col = num_arr_data[:, crt_idx]
        digitized_col = DiscretizeVect(crt_col, in_bin_no)
        data_digi_x[:, crt_idx] = digitized_col

    return data_digi_x

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def RandomShrinkTbl(in_large_data, in_reduced_no, in_random_seed = None) :

    out_data = []
    row_no = len(in_large_data)
    n_rows_to_remove = row_no - in_reduced_no

    if n_rows_to_remove > 0 :
        if not in_random_seed == None :
            np.random.seed(in_random_seed)  # Set the random seed
        indices_to_remove = list(np.random.choice(row_no, size=n_rows_to_remove, replace=False))
        indices_to_remove.sort(reverse=True)
        new_data = np.delete(in_large_data, indices_to_remove, axis=0)
        new_list_data = C4pTblUtil.ListDataConvert(new_data)
        return new_list_data
    else :
        return in_large_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def TransformTblTranslateElem( in_tbl_data, in_transl_dict = None) :

    if ( in_transl_dict == None or in_transl_dict == {} ) :
        return in_tbl_data

    row_no = len(in_tbl_data)
    col_no = len(in_tbl_data[0])
    new_tbl = []
    for crt_i_idx in range(row_no) :
        new_row = []
        for crt_j_idx in range(col_no) :
            crt_elem = in_tbl_data[crt_i_idx][crt_j_idx]
            if crt_elem in in_transl_dict :
                new_elem = in_transl_dict[crt_elem]
            else :
                new_elem = crt_elem
            new_row.append(new_elem)
        new_tbl.append(new_row)
    return new_tbl

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def AvgAccuracyDataTest(in_train_data, in_target_data, classif_entry, iterations = 1, random_seed = None, test_fraction = 0.5) :

    import sys
    import io

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    begin_time_ref = datetime.datetime.now()
    crt_time_ref = datetime.datetime.now()

    if classif_entry[1] == None :
        classif_id = str(classif_entry[0])
    else :
        classif_id = classif_entry[1]
    classifier = classif_entry[0]

    # disable_std_output = False
    disable_std_output = True

    # x_data = np.array(in_train_data)
    x_data = in_train_data
    y_target = in_target_data

    cumulate_acc = 0
    excpt_count = 0
    acc_list = []

    crt_rand_seed = random_seed
    for crt_idx in range(iterations) :
        if not random_seed == None :
            crt_rand_seed = random_seed + crt_idx
        ret_tuple = train_test_split(x_data, y_target, test_size = test_fraction, random_state = crt_rand_seed)
        x_train, x_test, y_train, y_test = ret_tuple

        if disable_std_output :
            save_stdout = sys.stdout
            save_stderr = sys.stderr
            stdout = io.StringIO()
            stderr = io.StringIO()
            sys.stdout = stdout
            sys.stderr = stderr
            output = stdout.getvalue()
            error = stderr.getvalue()

        new_exception_flag = False

        try :
            accuracy = -1
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)

        except Exception as excerr:
            excpt_count += 1
            if accuracy < 0 or accuracy > 1.0 :
                accuracy = 0.0
            new_exception_flag = True
            exc_type = str(type(excerr))
            exc_msg = str(excerr)

        if disable_std_output :
            sys.stdout = save_stdout
            sys.stderr = save_stderr

        if new_exception_flag :
            print("- exception - classif_id:", classif_id)
            print("    type: %s" % (exc_type[:70]))
            print("    err: %s" % (exc_msg[:71]))

        cumulate_acc += accuracy
        acc_list.append(accuracy)

    new_time_ref = datetime.datetime.now()
    delta = new_time_ref - crt_time_ref
    delta_secs = delta.total_seconds()
    crt_time_ref = new_time_ref
    if iterations > 1 :
        std_dev = statistics.stdev(acc_list)
    else :
        # std_dev = -1
        std_dev = 0
    avg_accuracy = (cumulate_acc * 1.0) / iterations
    return avg_accuracy, std_dev, [excpt_count, acc_list], delta_secs

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def ExecClassifAccuracyTest(x_data, y_data, classifier_lst, iter_no = 1, random_seed = None, test_fraction = 0.5, display_flag = True) :

    import time

    ref_time = time.time()
    test_tbl = []

    for crt_classif_entry in classifier_lst :
        if crt_classif_entry[1] == None :
            classif_id = str(crt_classif_entry[0])
        else :
            classif_id = crt_classif_entry[1]
        if display_flag: print("- - - - classif_id:", classif_id)
        accuracy, std_dev, extra_info, duration = AvgAccuracyDataTest(x_data, y_data, crt_classif_entry, iterations=iter_no, random_seed=random_seed, test_fraction=test_fraction)
        if display_flag: print("- - - - - - - - accuracy:", accuracy)
        if display_flag: print("- - - - - - - - std_dev:", std_dev)

        crt_time = time.time()
        time_delta = crt_time - ref_time
        ref_time = crt_time
        if display_flag: print("- - - - - - - - time_delta:", time_delta)
        if display_flag: C4pUseCommon.flush()

        test_tbl.append( [accuracy, str(classif_id), extra_info] )
    return test_tbl

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def GetRankScore(in_unsorted_list, reverse_order = False) :

    competitor_no = len(in_unsorted_list)
    if competitor_no == 1 :
        ret_rank_list = [0.5]
        return ret_rank_list
    zip_list = []
    for crt_idx in range(competitor_no) :
        crt_elem = in_unsorted_list[crt_idx]
        zip_list.append([crt_elem, crt_idx])

    sorted_list = sorted(zip_list, key=lambda row: row[0], reverse=reverse_order)
    rank_idx_list = [None] * competitor_no

    equal_len = 1
    cumulate_rank = 0
    for crt_idx in range(competitor_no) :
        crt_elem = sorted_list[crt_idx]
        if crt_idx == competitor_no - 1 :
            next_elem = ({}, None)
        else :
            next_elem = sorted_list[crt_idx+1]

        if crt_elem[0] == next_elem[0] :
            equal_len += 1
            cumulate_rank += (crt_idx)
        else :
            if equal_len > 1 :
                # identical run is broken
                cumulate_rank += (crt_idx)
                avg_rank = cumulate_rank/(equal_len*(competitor_no - 1.0))
                for crt_eq_idx in range(equal_len) :
                    update_idx = crt_idx - crt_eq_idx
                    rank_idx_list[update_idx] = [avg_rank, sorted_list[update_idx][1]]
            else :
                update_idx = crt_idx
                rank_idx_list[update_idx] = [update_idx/(competitor_no - 1.0), crt_elem[1]]
            equal_len = 1
            cumulate_rank = 0

    # now generate simple rank list ordered as input
    ret_rank_list = [None] * competitor_no
    for crt_idx in range(competitor_no) :
        crt_row = rank_idx_list[crt_idx]
        [rank_score, original_idx] = crt_row[:]
        ret_rank_list[original_idx] = rank_score
    return ret_rank_list

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def ScoreRankTbl(acc_class_iter_list, reverse_order = False) :

    acc_iter_class_list = C4pTblUtil.MatrixTranspose(acc_class_iter_list)
    rank_iter_class_list = []
    for crt_row in acc_iter_class_list :
        crt_rank_score_list = GetRankScore(crt_row, reverse_order)
        # crt_rank_score_list = crt_row
        rank_iter_class_list.append(crt_rank_score_list)
    rank_class_iter_list = C4pTblUtil.MatrixTranspose(rank_iter_class_list)

    rank_score_list = []
    for crt_row in rank_class_iter_list :
        crt_rank_avg = statistics.mean(crt_row)
        len_row = len(crt_row)
        if len_row > 1 :
            crt_rank_std = statistics.stdev(crt_row)
        else :
            crt_rank_std = 0
        rank_score_list.append([crt_rank_avg, crt_rank_std])

    rank_class_list = rank_score_list
    return rank_class_list

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def BatchClassifRankAccTest(x_data, y_data, classifier_lst, iter_no = 1, random_seed = None, test_fraction = 0.5, display_flag = True) :

    if display_flag: C4pUseCommon.CrtTimeStamp()
    if display_flag: print("- - - - - - - - - - - - ")
    classif_tbl = ExecClassifAccuracyTest(x_data, y_data, classifier_lst,
                                          iter_no, random_seed, test_fraction,
                                          display_flag)
    classif_no = len(classifier_lst)
    acc_class_iter_list = []
    for crt_row in classif_tbl :
        crt_extra_info = crt_row[2]
        acc_class_iter_list.append(crt_extra_info[1])

    rank_class_list = ScoreRankTbl(acc_class_iter_list, reverse_order = False)
    if display_flag: print("- - - - - - - - - - - - ")
    test_tbl = []
    for crt_idx in range(classif_no) :
        crt_classif_entry = classifier_lst[crt_idx]
        if crt_classif_entry[1] == None :
            classif_id = str(crt_classif_entry[0])
        else :
            classif_id = crt_classif_entry[1]
        crt_classif = classif_id
        accuracy, extra_info, duration = classif_tbl[crt_idx][0], classif_tbl[crt_idx][2], classif_tbl[crt_idx][1]
        rank_score_avg = rank_class_list[crt_idx][0]
        rank_score_stdev = rank_class_list[crt_idx][1]
        if display_flag: print("- - - - - - - - - - - - ")
        if display_flag: print("- - - - crt_classif:", crt_classif)
        # if display_flag: print("- - - - accuracy:", accuracy)
        if display_flag: print("- - - - rank_score_avg:", rank_score_avg)
        if display_flag: print("- - - - rank_score_stdev:", rank_score_stdev)
        if display_flag: print("- - - - excpt_count:", extra_info[0])
        # if display_flag: C4pUseCommon.CrtTimeStamp()
        new_row = [rank_score_avg, rank_score_stdev]
        new_row += [accuracy, crt_classif, extra_info, duration]
        test_tbl.append(new_row)

    if display_flag: print("- - - - - - - - - - - - ")
    if display_flag: C4pUseCommon.CrtTimeStamp()
    if display_flag: iprnt()
    return test_tbl

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def PlotFunctionList(func_list, interval):

    x = np.linspace(interval[0], interval[1], 100)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, func in enumerate(func_list):
        print("    crt_fn:", func)
        y = func(x)
        plt.plot(x, y, color=colors[i % len(colors)])
    plt.show()

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def TblSampleDisplay(table_list, display_flag = True) :

    data_no_row = len(table_list)
    data_no_col = len(table_list[0])

    display_sample_row = 5
    display_sample_col = 5
    min_sample_row = min(data_no_row, 2*display_sample_row)
    min_sample_col = min(data_no_col, 2*display_sample_col)

    dot_flag = False
    for crt_idx in range(data_no_row) :
        if (crt_idx < display_sample_row
            or crt_idx >= (data_no_row - display_sample_row)) :
            crt_row = table_list[crt_idx]
            if (2*display_sample_col < data_no_col) :
                str_begin = str(crt_row[:display_sample_col])
                str_end = str(crt_row[-display_sample_col:])
                if display_flag: print("    " + str_begin + " ... " + str_end)
            else :
                str_print = str(crt_row)
                if display_flag: print("    " + str_print)
        else :
            if not dot_flag :
                if display_flag: print("    ...")
                dot_flag = True
    return

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def BatchCsvAccuracyTest(predictor_list, file_data_list, data_location, iter_no = 3,
                            random_seed = 42, test_fraction = 0.5, data_process_mode = 'numeric',
                            array_limit_row_max = 10000, array_limit_row_min = 12,
                            array_limit_col_max = 0, display_flag = True) :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    csv_file_lst = file_data_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    force_head = None
    default_set_col = [-1, force_head]
    csv_set_lst = []
    for crt_row in csv_file_lst :
        crt_len = len(crt_row)
        concat_row = crt_row + default_set_col[(crt_len-1):]
        new_row = concat_row[:]
        csv_set_lst.append(new_row)
    data_set_lst = csv_set_lst

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    classifier_lst = predictor_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    random.seed(random_seed)
    str_vers = GetVersion()

    if display_flag: print("- - - - - - - - - ")
    if display_flag: print("- - - - - - - - - ")
    if display_flag: print("- - BatchCsvAccuracyTest (%s) - begin"%(str_vers))
    if display_flag: print("- - - - - - - - - ")
    if display_flag: print("- - - - batch average accuracy test")
    if display_flag: print()
    if display_flag: print("- - - - iter_no:", iter_no)
    if display_flag: print("- - - - random_seed:", random_seed)
    if display_flag: print("- - - - test_fraction:", test_fraction)
    if display_flag: print("- - - - data_process_mode:", data_process_mode)
    if display_flag: print()

    if display_flag: print("- - - - - - array_limit_row_max:", array_limit_row_max)
    if display_flag: print("- - - - - - array_limit_row_min:", array_limit_row_min)
    if display_flag: print("- - - - - - array_limit_col_max:", array_limit_col_max)
    if display_flag: print()

    agg_dataset_tbl = []
    classif_no = len(classifier_lst)
    if display_flag: print("- - - - - - - - - ")
    if display_flag: print("- - - classifier no:", classif_no)
    if display_flag: print("")
    if display_flag: RowListDisplay(classifier_lst)
    if display_flag: print("")

    for crt_idx in range(classif_no) :
        crt_classif_entry = classifier_lst[crt_idx]
        if crt_classif_entry[1] == None :
            classif_id = str(crt_classif_entry[0])
        else :
            classif_id = crt_classif_entry[1]
        crt_classif = classif_id
        agg_dataset_tbl.append([[], [], 0, crt_classif])

    dataset_no = len(data_set_lst)
    if display_flag: print("- - - - - - - - - ")
    if display_flag: print("- - - dataset_no:", dataset_no)
    if display_flag: print("")
    if display_flag: RowListDisplay(data_set_lst)

    for crt_idx in range(dataset_no) :
        crt_data_set = data_set_lst[crt_idx]
        desc_name = crt_data_set[0]
        if display_flag: print("- - - - - - - - - ")
        if display_flag: print("- - - - dataset:", desc_name)

        default_entry_list = [crt_data_set[0], -1, None, None]
        entry_param_list = FillMissing( default_entry_list, crt_data_set)

        input_csv = data_location + entry_param_list[0]
        input_targ_idx = entry_param_list[1]
        input_max_onehot = entry_param_list[2]
        input_head_first = entry_param_list[3]

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ret_data = RetrieveCsvTbl(
                                    input_csv,
                                    input_head_first,
                                   )
        input_train_data, insput_start_row = ret_data

        input_no_row = len(input_train_data)
        input_no_col = len(input_train_data[0])

        if display_flag: print("- - - - - - - - - ")
        if display_flag: print("- - - data processing - retrieved csv")
        if display_flag: print("- - - - - - input_no_row:", input_no_row)
        if display_flag: print("- - - - - - input_no_col:", input_no_col)
        if display_flag: print("- - - - - - input_targ_idx:", input_targ_idx)
        if display_flag: print("- - - - - - insput_start_row:", insput_start_row)
        if display_flag: print("- - - - - - - - - ")

        if display_flag: print("- - - data sample:")
        TblSampleDisplay(input_train_data, display_flag)

        if display_flag: print("- - - - - - - - - ")
        if display_flag: print()

        input_elem_no = input_no_row * input_no_col

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ret_data = TblTargetExtract(
                                    input_train_data,
                                    input_targ_idx,
                                   )
        y_target, attr_data_tbl = ret_data
        if display_flag: print("- - - data processing - target extracted")

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ret_data = TransformTblOneHotEncoding(
                                    attr_data_tbl,
                                    input_max_onehot,
                                   )
        proc_train_data, insert_list = ret_data

        proc_no_row = len(proc_train_data)
        proc_no_col = len(proc_train_data[0])

        if not proc_no_row == input_no_row :
            if display_flag: print(" Error - inconsistent processing")
            return None

        onehot_no_row = proc_no_row
        onehot_no_col = proc_no_col

        if display_flag: print("- - - data processing - one-hot encoded")
        if display_flag: print("- - - - - - onehot_no_row:", onehot_no_row)
        if display_flag: print("- - - - - - onehot_no_col:", onehot_no_col)

        # if display_flag: print("- - - proc_train_data:")

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ret_data = TransformTblDataSizeLimit(proc_train_data, y_target,
                        array_limit_row_max, array_limit_row_min,
                        array_limit_col_max)

        proc_train_data, y_target = ret_data

        proc_no_row = len(proc_train_data)
        proc_no_col = len(proc_train_data[0])

        if display_flag: print("- - - data processing - size limitation")
        if display_flag: print("- - - - - - proc_no_row:", proc_no_row)
        if display_flag: print("- - - - - - proc_no_col:", proc_no_col)

        if proc_no_row == onehot_no_row :
            if display_flag: print("- - - data row number unchanged")
        elif proc_no_row < onehot_no_row :
            if display_flag: print("- - - data row number reduced")
        else :
            if display_flag: print("- - - data row number augmented")

        if (proc_no_col) == onehot_no_col :
            if display_flag: print("- - - data col number unchanged")
        elif (proc_no_col) < onehot_no_col :
            if display_flag: print("- - - data col number reduced")
        else :
            if display_flag: print("- - - data col number augmented")

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if data_process_mode == 'categ_sim_a' :

            # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Simulate categorical input by discretizing numbers into 0 and 1

            discretize_param = 2

            new_proc_data = TransformTblDiscretize(proc_train_data, discretize_param)
            proc_train_data = new_proc_data

            proc_no_row = len(proc_train_data)
            proc_no_col = len(proc_train_data[0])

            if display_flag: print("- - - data processing - discretization")
            if display_flag: print("- - - - - - proc_no_row:", proc_no_row)
            if display_flag: print("- - - - - - proc_no_col:", proc_no_col)

            # > - - - - - - - - - - - - - - - - - - - - - - - - - - -

        elif data_process_mode == 'categ_sim_b' :

            # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Simulate categorical input by discretizing numbers into four symbols
            # then transforming them back into numbers using one-hot encoding

            discretize_param = 4

            new_proc_data = TransformTblDiscretize(proc_train_data, discretize_param)
            proc_train_data = new_proc_data

            proc_no_row = len(proc_train_data)
            proc_no_col = len(proc_train_data[0])

            if display_flag: print("- - - data processing - discretization")
            if display_flag: print("- - - - - - proc_no_row:", proc_no_row)
            if display_flag: print("- - - - - - proc_no_col:", proc_no_col)

            # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Translate numerical elements

            # translate_dict = None
            translate_dict = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e'}
            new_proc_data = TransformTblTranslateElem(proc_train_data, translate_dict)
            proc_train_data = new_proc_data

            if display_flag: print("- - - data processing - element translation")

            # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ret_data = TransformTblOneHotEncoding(
                                        proc_train_data,
                                        5,
                                       )
            proc_train_data, insert_list = ret_data

            proc_no_row = len(proc_train_data)
            proc_no_col = len(proc_train_data[0])

            if display_flag: print("- - - data processing - one-hot encoded")
            # > - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if display_flag: print("- - - data processing - finished")
        if display_flag: print("- - - - - - proc_no_row:", proc_no_row)
        if display_flag: print("- - - - - - proc_no_col:", proc_no_col)
        if display_flag: print("- - - data sample:")
        TblSampleDisplay(proc_train_data, display_flag)

        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # > - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if display_flag: print()
        if display_flag: C4pUseCommon.SepLine2()

        test_tbl = BatchClassifRankAccTest(proc_train_data,
                                            y_target, classifier_lst,
                                            iter_no = iter_no,
                                            random_seed = random_seed,
                                            test_fraction = test_fraction,
                                            display_flag = display_flag)

        # aggregate rank data
        for crt_idx in range(classif_no) :
            crt_agg_row = agg_dataset_tbl[crt_idx]
            crt_test_row = test_tbl[crt_idx]
            row_avg_rank = crt_test_row[0]
            row_avg_acc = crt_test_row[2]
            row_except = crt_test_row[4][0]
            crt_agg_row[0].append(row_avg_rank)
            crt_agg_row[1].append(row_avg_acc)
            crt_agg_row[2] += row_except

        rank_classif = reversed(sorted(test_tbl, key=lambda row: row[0]))
        # print_tab_dist = [20, 20, 8]
        print_tab_dist = [20, 7, 20, 5]

        if display_flag: C4pUseCommon.SepLine2()
        if display_flag: print("- - - - dataset name:", desc_name)
        if display_flag: print("")
        line_str = C4pUseCommon.ListToTabStr(["rank score avg", "stdev", "accuracy avg", "xpt", "classifier"], print_tab_dist)
        if display_flag: print(line_str)
        if display_flag: C4pUseCommon.SepLine2()
        for crt_entry in rank_classif :
            row_avg_rank = crt_entry[0]
            row_std_rank = crt_entry[1]
            row_avg_acc = crt_entry[2]
            row_except = crt_entry[4][0]
            row_classif = crt_entry[3]
            # line_str = C4pUseCommon.ListToTabStr([crt_entry[0], crt_entry[1], crt_entry[3][0], crt_entry[2]], print_tab_dist)
            line_str = C4pUseCommon.ListToTabStr([row_avg_rank, row_std_rank, row_avg_acc, row_except, row_classif], print_tab_dist)
            if display_flag: print(line_str)
        if display_flag: C4pUseCommon.SepLine2()
        if display_flag: print()

    unsorted_rank_list =[]
    for crt_idx in range(classif_no) :
        crt_row = agg_dataset_tbl[crt_idx]
        crt_rank_list = crt_row[0]
        crt_acc_list = crt_row[1]
        row_except = crt_row[2]
        row_classif = crt_row[3]
        agg_rank_avg = statistics.mean(crt_rank_list)
        agg_acc_avg = statistics.mean(crt_acc_list)
        crt_rank_no = len(crt_rank_list)
        if crt_rank_no > 1 :
            agg_rank_stdev = statistics.stdev(crt_rank_list)
            agg_acc_stdev = statistics.stdev(crt_acc_list)
        else :
            agg_rank_stdev = 0
            agg_acc_stdev = 0
        unsorted_rank_list.append([agg_rank_avg, agg_rank_stdev, agg_acc_avg, agg_acc_stdev, row_except, row_classif])

    if display_flag: print("- - - - - - - - - ")
    if display_flag: print()

    agg_rank_classif = sorted(unsorted_rank_list, key=lambda row: row[2])
    agg_rank_classif.reverse()

    # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print("- - - - aggregate accuracy")
    if display_flag: print("")
    print_tab_dist = [4, 20, 10, 8]
    line_str = C4pUseCommon.ListToTabStr(["no", "accuracy avg", "stddev", "excpt", "classifier"], print_tab_dist)
    if display_flag: print(line_str)
    if display_flag: C4pUseCommon.SepLine2()

    agg_value_list = []
    agg_stdev_list = []

    agg_rank_no = len(agg_rank_classif)
    # for crt_entry in agg_rank_classif :
    for crt_idx in range(agg_rank_no) :
        crt_entry = agg_rank_classif[crt_idx]
        row_avg_rank = crt_entry[0]
        row_std_rank = crt_entry[1]
        row_avg_acc = crt_entry[2]
        row_std_acc = crt_entry[3]
        row_except = crt_entry[4]
        row_classif = crt_entry[5]
        line_str = C4pUseCommon.ListToTabStr([(crt_idx+1), row_avg_acc, row_std_acc, row_except, row_classif], print_tab_dist)
        if display_flag: print(line_str)
        agg_value_list.append(row_avg_acc)
        agg_stdev_list.append(row_std_acc)
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print()

    if agg_rank_no > 0 :
        avg_value = statistics.mean(agg_value_list)
        avg_stdev = statistics.mean(agg_stdev_list)
    else :
        avg_value = 0
        avg_stdev = 0

    if display_flag: print("    avg_value:", avg_value)
    if display_flag: print("    avg_stdev:", avg_stdev)
    if display_flag: print()
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print()

    # > - - - - - - - - - - - - - - - - - - - - - - - - - - -

    agg_rank_classif = sorted(unsorted_rank_list, key=lambda row: row[0])
    agg_rank_classif.reverse()

    # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print("- - - - aggregate rank score")
    if display_flag: print("")
    print_tab_dist = [4, 20, 10, 8]
    line_str = C4pUseCommon.ListToTabStr(["no", "rank aggregate avg", "stddev", "excpt", "classifier"], print_tab_dist)
    if display_flag: print(line_str)
    if display_flag: C4pUseCommon.SepLine2()

    agg_value_list = []
    agg_stdev_list = []

    agg_rank_no = len(agg_rank_classif)
    # for crt_entry in agg_rank_classif :
    for crt_idx in range(agg_rank_no) :
        crt_entry = agg_rank_classif[crt_idx]
        row_avg_rank = crt_entry[0]
        row_std_rank = crt_entry[1]
        row_avg_acc = crt_entry[2]
        row_std_acc = crt_entry[3]
        row_except = crt_entry[4]
        row_classif = crt_entry[5]
        line_str = C4pUseCommon.ListToTabStr([(crt_idx+1), row_avg_rank, row_std_rank, row_except, row_classif], print_tab_dist)
        if display_flag: print(line_str)
        agg_value_list.append(row_avg_rank)
        agg_stdev_list.append(row_std_rank)
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print()

    if agg_rank_no > 0 :
        avg_value = statistics.mean(agg_value_list)
        avg_stdev = statistics.mean(agg_stdev_list)
    else :
        avg_value = 0
        avg_stdev = 0

    if display_flag: print("    avg_value:", avg_value)
    if display_flag: print("    avg_stdev:", avg_stdev)
    if display_flag: print()
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print()
    if display_flag: print("- - BatchCsvAccuracyTest (%s) - end"%(str_vers))
    if display_flag: print()

    # > - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if display_flag: C4pUseCommon.SepLine2()
    if display_flag: print()

    #""" # comment - end
    return rank_classif

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
