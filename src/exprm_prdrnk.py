"""
    Module used in developing, experimenting and testing.

        Tested with Winpython64-3.10.5.0
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

import usap_common

# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------
# from usap_common import *

iprnt = usap_common.iprnt

# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------

utest_test_no = 0
utest_fail_counter = 0

# >-----------------------------------------------------------------------------

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def file_arff_to_csv(in_filename) :

    import os
    import csv

    def convert_arff_to_csv(arff_file_path):
        # Get the base name of the ARFF file
        base_name = os.path.splitext(arff_file_path)[0]
        csv_file_path = f"{base_name}.csv"

        with open(arff_file_path, 'r') as f:
            # Skip the @relation line
            next(f)

            # Get the attribute lines and save them in a list
            attributes = []
            line = next(f).strip()
            while line.startswith('@attribute'):
                attributes.append(line.split()[1].strip("'"))
                line = next(f).strip()

            # Skip the @data line
            next(f)

            # Read the data lines
            data = []
            for line in f:
                values = line.strip().split(',')
                data.append(values)

            # Write the data to a CSV file
            with open(csv_file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(attributes)
                writer.writerows(data)

    print("- - - in_filename:", in_filename)

    # Specify the ARFF file
    # arff_file = 'your_file.arff'
    arff_file = in_filename

    # Convert the ARFF file to CSV
    convert_arff_to_csv(arff_file)

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def reduce_csv_file(filename, reduction_percentage):

    import pandas as pd
    import numpy as np

    # Read the CSV file
    df = pd.read_csv(filename)

    # Calculate the number of rows to remove
    n_rows_to_remove = int(len(df) * reduction_percentage / 100)

    # Randomly select rows to remove
    indices_to_remove = np.random.choice(df.index, size=n_rows_to_remove, replace=False)

    # Remove the selected rows
    df_reduced = df.drop(indices_to_remove)

    # Create a new filename with an appendix
    basename, extension = filename.rsplit('.', 1)
    new_filename = f"{basename}-reduced.{extension}"

    # Save the reduced DataFrame as a new CSV file
    df_reduced.to_csv(new_filename, index=False)

    print(f"- - - - - - - - reduced file: {new_filename}")

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def batch_arff_to_csv(in_dir_arff = '.') :

    import os

    # Get all files in the directory
    files = os.listdir(in_dir_arff)

    # Filter the list to include only .arff files
    arff_files = [f for f in files if f.endswith('.arff')]
    no_files = len(arff_files)
    print ("- - - - no_files no:", no_files)
    print ()

    # Convert each ARFF file to CSV
    for arff_file in arff_files:
        file_path = os.path.join(in_dir_arff, arff_file)
        file_arff_to_csv(file_path)

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def batch_reduce_csv() :

    reduce_tbl = [
            ["../misc/dataset/delgado/all-csv/miniboone--miniboone.csv", 95],
            ["../misc/dataset/delgado/all-csv/connect-4--connect-4.csv", 90],
            ["../misc/dataset/delgado/all-csv/musk-2--musk-2.csv", 60],
            ["data/adult.csv", 50],
        ]

    reduce_tbl = [
            ["data/dtest/delgado/adult--adult_train.csv", 43.785712936571397],
            ["data/dtest/delgado/musk-2--musk-2-reduced.csv", 37.764144529580324],
            ["data/dtest/delgado/statlog-shuttle--statlog-shuttle_train.csv", 37.59818924921636],
            ["data/dtest/delgado/semeion--semeion.csv", 33.71617105363979],
            ["data/dtest/delgado/miniboone--miniboone-reduced.csv", 22.447771676313133],
            ["data/dtest/delgado/letter--letter.csv", 17.217693294202685],
            ["data/dtest/delgado/connect-4--connect-4-reduced.csv", 10.176660950309369],
        ]

    for crt_row in reduce_tbl :
        print ("- - - - crt_row:", crt_row)
        reduce_csv_file(crt_row[0], crt_row[1])
        print()

    print()
    print("- - - - reduction done")
    print()

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------
# > Begin C4setExprmnt
# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------

class C4setExprmnt :

    # >-----------------------------------------------------------------------------
    # >-----------------------------------------------------------------------------

    sysDebug = 3

    globalDef = {}
    globalDef['cache_dict'] = {}
    globalDef['profiling'] = {}
    # globalDef['profiling']['enable'] = False
    globalDef['profiling']['enable'] = True

    import time
    crt_time = time.time()
    globalDef['profiling']['general_start'] = crt_time
    globalDef['profiling']['tags'] = {}

    globalDef['konst'] = {}
    globalDef['var'] = {}

    # >-----------------------------------------------------------------------------

    @staticmethod
    def SysSep ( display_flag = True ) :
        separator_string = ">-------------------------------------------------------------------------------"
        if display_flag : print (separator_string)
        return separator_string
    # >-----------------------------------------------------------------------------

    @staticmethod
    def SysProfileStart( in_tag_name ) :
        """
        Start tag reference

        Params:
            in_tag_name

        Returns:
            ret_status
        """
        if C4setExprmnt.sysDebug > 3.107 :
            C4setExprmnt.SysLogMsg("trace")
            print ("cadbg2 - param, in_tag_name", in_tag_name)

        # enabling_flag, general_start = C4setExprmnt.Misc.SysProfileGetStatus()
        enabling_flag = C4setExprmnt.globalDef['profiling']['enable']
        if not enabling_flag :
            return False

        import time
        crt_time = time.time()
        # profiling_dict = C4setExprmnt.globalDef['profiling']['tags'][in_tag_name]
        profiling_dict = C4setExprmnt.globalDef['profiling']['tags']

        if not in_tag_name in profiling_dict :
            profiling_dict[in_tag_name] = {'cumulate': 0, 'start': crt_time}
        else :
            profiling_dict[in_tag_name]['start'] = crt_time

        return True
    # >-----------------------------------------------------------------------------

    @staticmethod
    def SysProfileStop( in_tag_name ) :
        """
        Cumulate tag reference

        Params:
            in_tag_name

        Returns:
            ret_status

        """
        if C4setExprmnt.sysDebug > 3.107 :
            C4setExprmnt.SysLogMsg("trace")
            print ("cadbg2 - param, in_tag_name", in_tag_name)

        # enabling_flag, general_start = C4setExprmnt.Misc.SysProfileGetStatus()
        enabling_flag = C4setExprmnt.globalDef['profiling']['enable']
        if not enabling_flag :
            return False

        import time
        crt_time = time.time()
        profiling_dict = C4setExprmnt.globalDef['profiling']['tags'][in_tag_name]
        start_time = profiling_dict['start']
        delta = crt_time - start_time
        profiling_dict['cumulate'] += delta

        return True
    # >-----------------------------------------------------------------------------

    @staticmethod
    def SysProfileGetStatus() :
        """
        Reset profiling data

        Params:
        Returns:
            ret_enable_status
            ret_general_start
        """
        if C4setExprmnt.sysDebug > 3.107 :
            C4setExprmnt.SysLogMsg("trace")
            print ("cadbg2 - param, in_tag_name", in_tag_name)

        enabling_flag = C4setExprmnt.globalDef['profiling']['enable']
        general_start = C4setExprmnt.globalDef['profiling']['general_start']
        return enabling_flag, general_start
    # >-----------------------------------------------------------------------------

    @staticmethod
    def SysProfileReset(enable_flag = None) :
        """
        Reset profiling data

        Params:
            enable_flag
        Returns:
            ret_status
        """
        if C4setExprmnt.sysDebug > 3.107 :
            C4setExprmnt.SysLogMsg("trace")
            print ("cadbg2 - param, in_tag_name", in_tag_name)

        import time
        crt_time = time.time()

        if not enable_flag == None :
            C4setExprmnt.globalDef['profiling']['enable'] = enable_flag

        C4setExprmnt.globalDef['profiling']['general_start'] = crt_time
        C4setExprmnt.globalDef['profiling']['tags'] = {}

        return True
    # >-----------------------------------------------------------------------------

    @staticmethod
    def SysProfileStats( display_flag = True ) :
        """
        Display profiling data

        Param:
            display_flag

        Return:
            ret_status
            ret_stats_data
            ret_stats_str

        """

        if C4setExprmnt.sysDebug > 3.107 :
            C4setExprmnt.SysLogMsg("trace")
            print ("cadbg2 - param, display_flag", display_flag)

        import time
        crt_time = time.time()

        fn_ret_status = False
        fn_ret_stats_data = None
        fn_ret_stats_str = None

        # one iteration loop to allow unified return through loop breaks
        for dummy_idx in range(1) :

            profile_dict = C4setExprmnt.globalDef['profiling']
            gen_start = profile_dict['general_start']
            gen_elapsed = crt_time - gen_start

            out_str_buf = []
            profile_stats_data = []

            crt_str = C4setExprmnt.SysSep()
            out_str_buf.append(crt_str)
            if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            crt_str = str( "  Profiling stats" % () )
            out_str_buf.append(crt_str)
            if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            table_separator = str( "    %s" % (50*'-') )
            crt_str = table_separator
            out_str_buf.append(crt_str)
            if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            profile_keylist = list(profile_dict['tags'].keys())
            print("experiment - profile_keylist:", profile_keylist)
            profile_keylist.sort()
            # for crt_profile_key in profile_dict :
            for crt_profile_key in profile_keylist :

                profile_tag_dict = profile_dict['tags'][crt_profile_key]
                crt_cumulate = profile_tag_dict['cumulate']
                epsilon_val = 1e-12
                crt_ratio = ((crt_cumulate + epsilon_val)* 100.0) / (gen_elapsed + epsilon_val)

                profile_stats_data.append( {'tag': crt_profile_key, 'cumulate': crt_cumulate, 'percent': crt_ratio} )

                # crt_str = str( "    tag: %-5d total: %-5d  ratio: %-5.3f      function: %-s" % ( profile_hit, profile_total, hit_ratio, fn_name ) )
                # crt_str = str( "    cumulate:  %-11.7f  ratio:  %-11.7f %%  tag: %s" % ( crt_cumulate, crt_ratio, crt_profile_key ) )
                crt_str = str( "    cumulate:  %-15.7f  ratio:  %-11.7f %%  tag: %s" % ( crt_cumulate, crt_ratio, crt_profile_key ) )
                out_str_buf.append(crt_str)
                if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            crt_str = table_separator
            out_str_buf.append(crt_str)
            if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            crt_str = str( "    general cumulate: %f" % ( gen_elapsed ) )
            out_str_buf.append(crt_str)
            if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            crt_str = C4setExprmnt.SysSep()
            out_str_buf.append(crt_str)
            if C4setExprmnt.sysDebug > 5.014585 : print ("cadbg5 - SysProfileStats, str: %s" % (crt_str))

            if display_flag :
                for crt_row in out_str_buf :
                    print (crt_row)

            fn_ret_status = True
            fn_ret_stats_data = profile_stats_data
            fn_ret_stats_str = out_str_buf
            break

        return fn_ret_status, fn_ret_stats_data, fn_ret_stats_str
    # >-----------------------------------------------------------------------------

# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------
# > End C4setExprmnt
# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------



# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# if __name__ == "__main__":
for dummy_idx in range(1) :

    C4setExprmnt.SysProfileReset()

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    usap_common.SepLine()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    usap_common.CrtTimeStamp()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    usap_common.SepLine()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    C4setExprmnt.SysProfileStart('tag_prdrnk_devtest_main')

    # >- - - - - - - - - - -
    iprnt()
    iprnt("- - - - - - - - - ")
    iprnt("- - - - prdrnk_devtest - start")
    iprnt("- - - - - - - - - ")
    iprnt()

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    import prdrnk_devtest

    prdrnk_devtest.predirank_devtest()
    iprnt()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # >- - - - - - - - - - -
    usap_common.SepLine()
    iprnt ("- - - Aggregate results")
    iprnt ("- - -   utest_test_no:", utest_test_no)
    iprnt ("- - -   utest_fail_counter:", utest_fail_counter)
    if utest_fail_counter == 0:
        iprnt ("- - -   UnitTest succes")
    else:
        iprnt ("- - -   UnitTest failed !")
        iprnt ("- - -       errors:", utest_fail_counter)
    usap_common.SepLine()
    # >- - - - - - - - - - -
    
    iprnt()
    iprnt("- - - - - - - - - ")
    iprnt("- - - - - - - - - ")
    iprnt()

    C4setExprmnt.SysProfileStop('tag_prdrnk_devtest_main')

    iprnt("- - - - - - - - - ")
    iprnt()
    C4setExprmnt.SysProfileStats()

    iprnt("- - - - - - - - - ")
    iprnt("- - - - prdrnk_devtest - stop")
    iprnt()

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    usap_common.SepLine()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    usap_common.CrtTimeStamp()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    usap_common.SepLine()
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print()
    break

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
