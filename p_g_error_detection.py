import pandas as pd
import numpy as np
import kh_config as config


class ErrorDetection():
    def __init__(self, data_filename, columns_description_filename):

        self.df_cols_desc = pd.read_excel(columns_description_filename, sheetname='col_type')

        # get column to type map
        converters = {}
        self.colTypes = {}
        for row in self.df_cols_desc.iterrows():
            col = row[1]['FIELD_NAME']
            type_ = row[1]['TYPE']
            self.colTypes[col] = type_

            if type_ in ['PRIMARY_KEY', 'FOREIGN_KEY', 'ENUM', 'TEXT']:
                converters[col] = str

        # self.df_complete = pd.read_excel(data_filename, sheetname='data_original', na_values=" ")
        self.df_complete = pd.read_excel(data_filename, na_values=[" ", "NULL"])
        # self.df_complete = pd.read_excel(data_filename, sheetname='data', na_values=" ")   ##### sample data size 1000
        ### so the na_values allow you to defien what values are to be regarded as NA
        # self.df_complete = pd.read_excel(data_filename, sheetname='data_original', na_values=" ")

        self.unused_cols = list(self.df_cols_desc[self.df_cols_desc['UNUSED'] == True]['FIELD_NAME'])
        self.used_cols = list(self.df_cols_desc[self.df_cols_desc['UNUSED'] != True]['FIELD_NAME'])

        self.materials = list(self.df_complete['Material'].unique())
        self.plants = list(self.df_complete['Plant'].unique())
        self.materialTypes = list(self.df_complete['Material Type'].unique())

        # for col in self.df_complete.columns.values:
        #     self.df_complete[col] = self.df_complete[col].replace(" ", "")

        # print 'this is the value: "%s"' % self.df_complete[self.df_complete['MRP group'] == " "]['MRP group'][0]
        # print 'this is the value: "%s"' % self.df_complete['MRP group'][0]

        self.numeric_used_cols = []
        for col in self.used_cols:
            if self.colTypes[col] in ['INT', 'FLOAT']:
                self.numeric_used_cols += [col]

        # remove all cols that have only a single value
        self.colNumValues = []
        for col in self.used_cols:
            values = self.df_complete[col].unique()
            if len(values) == 1:
                self.unused_cols += [col]
                self.used_cols.remove(col)
            else:
                self.colNumValues += {(len(values), col)}
        self.colNumValues.sort()

        # remove unused columns
        self.df = self.df_complete[self.used_cols]

        # initialize suspicious values
        self.errors = []

    def writeCleanedData(self, filename):
        # save data as csv-file
        self.df.to_csv(filename, sep='\t', encoding='utf-8', index=False, columns=self.used_cols)

    def getValueCounter(self, values):

        valueCounter = {}
        total = len(values)
        for value in values:
            # print value
            if pd.isnull(value):
                valueCounter[np.nan] = valueCounter.get(np.nan, 0) + 1
            else:
                valueCounter[value] = valueCounter.get(value, 0) + 1
        return (valueCounter, total)

    def getValueCounterList(self, values):

        valueCounter, total = self.getValueCounter(values)

        freqValues = []
        for k, v in valueCounter.items():
            freqValues += [(v, k)]   ### original
            # freqValues += [(k, v)]
        freqValues.sort()

        return (freqValues, total)

# data_filename = config.data_file
# columns_description_filename = config.column_description_file
#
# ed = ErrorDetection(data_filename, columns_description_filename)
#
# print 'cols'
# print ed.used_cols
