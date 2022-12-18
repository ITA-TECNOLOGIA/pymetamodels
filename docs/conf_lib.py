#!/usr/bin/python3

import os, sys
import xlrd
import shutil

class conf_lib(object):

    """
        .. _conf_lib:

        **conf_lib for Sphinx**

        :platform: Unix, Windows
        :synopsis: Generator of sphinx rst
        :author: FLC

        :Dependences: None

    """

    def __init__(self):

        self.build = None
        if self._is_html(): self.build = "html"
        if self._is_latex(): self.build = "latex"
        print(sys.argv)
        pass

    def log(self,txt):

        print(txt)

    def error(self,txt):

        print(txt)
        error

    def theme_version(self,a,b,c):

        _version_info = (a,b,c)
        _version = ".".join(map(str, _version_info))

        return _version

    def _is_html(self):
        if "html" in sys.argv:
            return True
        if "HTML" in sys.argv:
            return True
        return False

    def _is_latex(self):
        if "latex" in sys.argv:
            return True
        if "LATEX" in sys.argv:
            return True
        return False

    def add_file(self, __file_name, __folder_origin, __folder_output, current_directory):

        """

        **Synopsis:**
            * Add files from origin to output

        :Dependences: None

        .. _add_bibligraphy_Mend:

        **Args:**
            * __file_name:
            * __folder_origin:
            * __folder_output:

        **Returns:**
            * List with value to add to bibtex_bibfiles

        **Error:**
            * None

        .. note::

            * None

        """

        _file_name = __file_name
        _folder_origin = __folder_origin
        _folder_output = __folder_output
        docs_originfolder = os.path.join(os.path.join(current_directory, os.pardir), _folder_origin)
        docs_outputfolder = os.path.join(os.path.join(current_directory, os.pardir), _folder_output)
        if not os.path.exists(docs_outputfolder): os.makedirs(docs_outputfolder)

        or_file = os.path.join(docs_originfolder, _file_name)
        dst_file = os.path.join(docs_outputfolder, _file_name)
        if os.path.isfile(or_file):
            shutil.copy(or_file, dst_file)

    def add_bibligraphy_Mend(self, bib_name_file, bib_folder, current_directory):

        """

        **Synopsis:**
            * Add bibligraphy from mendeley

        :Dependences: None

        .. _add_bibligraphy_Mend:

        **Args:**
            * bib_name_file:
            * bib_folder:
            * current_directory:

        **Returns:**
            * List with value to add to bibtex_bibfiles

        **Error:**
            * None

        .. note::

            * None

        """

        bib_src_file = os.path.join(bib_folder, bib_name_file)
        bib_dst_file = os.path.join(current_directory, bib_name_file)

        if os.path.isfile(bib_src_file):
            shutil.copy(bib_src_file, bib_dst_file)
            return [bib_name_file]
        else:
            return []

    def parse_tables_files(self, path_xls, sheet, tables_dir, data_dir):

        """

        **Synopsis:**
            * Parse table data with tables information

        :Dependences: None

        .. _parse_tables_files:

        **Args:**
            * path_xls:
            * sheet:

        **Returns:**
            * None

        **Error:**
            * None

        .. note::

            * None

        """

        (_rows, _columns) = (None, None)
        (rows, columns, lst) = self.parse_table_xls_to_list_table_read_xls(path_xls, sheet, _rows, _columns)

        lst_header = lst[0][:]

        for i in range(2,rows):
            ele = {}
            for j in range(columns):
                ele[lst_header[j]] = lst[i][j]

            if ele["_Type"] == "table":


                file_name = ele["file_name"]
                tbl_sheet = ele["tbl_sheet"]
                tbl_name = tbl_sheet
                headers = ele["headers"]
                tabularcolumns = ele["tabularcolumns"]
                caption = ele["caption"]
                longtable = ele["long_table"]

                output_path = os.path.join(tables_dir, tbl_name + ".inc")
                path_xls = os.path.join(data_dir, file_name + ".xls")
                self.parse_table_xls_to_list_table(path_xls, output_path, caption = caption, \
                    sheet = tbl_sheet, rows = None, columns = None, headers = headers, \
                    anchor = tbl_name, tabularcolumns = tabularcolumns, longtable = longtable)

            else:
                self.log("Not such type")

    def parse_figures_files_table(self, path_xls, sheet, images_dir):

        """

        **Synopsis:**
            * Parse table data and execute multiple parse_figures_files

        :Dependences: None

        .. _parse_figures_files:

        **Args:**
            * path_xls:
            * sheet:

        **Returns:**
            * None

        **Error:**
            * None

        .. note::

            * None

        """

        (_rows, _columns) = (None, None)
        (rows, columns, lst) = self.parse_table_xls_to_list_table_read_xls(path_xls, sheet, _rows, _columns)

        lst_header = lst[0][:]

        for i in range(2,rows):
            ele = {}
            for j in range(columns):
                ele[lst_header[j]] = lst[i][j]

            if ele["_Type"] == "figure":

                _name_img = ele["img_name"]
                _ext = ele["ext"]
                path_img = os.path.join(ele["raw_folder"],_name_img +_ext)
                output_path = os.path.join(images_dir,_name_img + ".inc")
                scale = "%i" % ele["scale"] + " %"
                scale_latex = "%i" % ele["scale_latex"] + " %"
                caption = ele["caption"]
                align = ele["align"]
                anchor = _name_img

                self.parse_figures_files(path_img, output_path, scale = scale, scale_latex = scale_latex, caption = caption, align = align, anchor = anchor)

            else:
                self.log("Not such type")


    def parse_figures_files(self, path_img, output_path, scale = r"100 %", scale_latex = r"100%", caption = None, \
        align = "center", _class = "parse-figure", anchor = None):

        """

        **Synopsis:**
            * Parse images into rst files

        :Dependences: None

        .. _parse_figures_files:

        **Args:**
            * path_img:
            * output_path:

        **Returns:**
            * Formated text

        **Error:**
            * None

        .. note::

            * None

        """

        align = "center"

        # Give the location of the file
        sep0 = "    "
        sep = "    "

        with open(output_path, 'w', encoding = 'utf-8') as f:
            f.write("\n")

            #####
            f.write("\n")
            if anchor is not None:
                f.write(".. _%s-ach:\n" % anchor)

            f.write(sep0+"\n")
            f.write(sep0+".. figure:: %s\n" % path_img)
            f.write(sep0+sep + ":alt: %s\n" % _class)
            f.write(sep0+sep + ":name: %s\n" % anchor)
            if align is not None:
                f.write(sep0+sep + ":align: %s\n" % align)
            f.write(sep0+sep + ":figclass: %s\n" % _class)
            if self.build == "html":
                if scale is not None:
                    f.write(sep0+sep  + ":scale: %s\n" % scale)
            if self.build == "latex":
                #f.write(sep + ":figwidth: %s\n" % scale_latex)
                #f.write(sep + ":width: %s\n" % scale_latex)
                if scale_latex is not None:
                    f.write(sep0+sep  + ":scale: %s\n" % scale_latex)
            if caption is not None:
                f.write(sep0+sep + " \n")
                f.write(sep0+sep + "%s\n" % caption)
                f.write(sep0+sep + " \n")

            f.write(sep0+".. raw:: latex\n\n")
            f.write(sep0+sep + "\\FloatBarrier\n")

            f.write(sep0+sep + " \n")

        f.close()

        pass

    def parse_table_xls_to_list_table(self, path_xls, output_path, caption = None, sheet = None, rows = None, \
        columns = None, headers = 1, tabularcolumns = None, widths = None, _class = "parse-table", anchor = None, \
        longtable = False):

        """

        **Synopsis:**
            * Read an |Excel| sheet and generate an list_table

        :Dependences: None

        .. _parse_table_xls_to_list_table:

        **Args:**
            * path_xls:
            * output_path:

        **Returns:**
            * Formated text

        **Error:**
            * None

        .. note::

            * None

        """

        lst_data = self.parse_table_xls_to_list_table_read_xls(path_xls, sheet, rows, columns)

        self.parse_table_xls_to_list_table_create(lst_data, output_path, caption, headers, tabularcolumns, widths, _class, anchor, longtable)

        pass

    def parse_table_xls_to_list_table_create(self, lst_data, output_path, caption, headers, tabularcolumns, widths, _class, anchor, longtable):

        # Get data
        (rows, columns, lstt) = lst_data

        align = "center"

        # Give the location of the file
        sep = "    "
        print(sys.version)
        with open(output_path, 'w', encoding = 'utf-8') as f:

            f.write("\n")
            if anchor is not None:
                f.write(".. _%s-ach:\n" % anchor)
                f.write("\n")
            if tabularcolumns is not None:
                if bool(longtable):
                    f.write(".. tabularcolumns:: %s\n" % tabularcolumns)
                else:
                    f.write(".. tabularcolumns:: %s\n" % tabularcolumns)
            if caption is None:
                f.write(".. list-table::\n")
            else:
                f.write(".. list-table:: %s\n" % caption)
            if anchor is not None:
                f.write(sep + ":name: %s\n" % anchor)
            if headers is not None:
                f.write(sep + ":header-rows: %i\n" % headers)
            if widths is None:
                if bool(longtable):
                    if self.build == "latex":
                        f.write(sep + ":widths: %s\n" % ("1"+" 1"*(int(columns)-1)))
                    if self.build == "html":
                        f.write(sep + ":widths: auto\n")
                else:
                    f.write(sep + ":widths: auto\n")
            else:
                f.write(sep + ":widths: %s\n" % widths)
            if align is not None:
                f.write(sep + ":align: %s\n" % align)
            f.write(sep + r":width: 100%" + "\n")
            if bool(longtable):
                f.write(sep + r":class: longtable right-align-right-col align-cols left-align-left-col %s" % _class + "\n")
            else:
                f.write(sep + r":class: right-align-right-col align-cols left-align-left-col %s" % _class + "\n")

            f.write(sep + "\n")

            for i in range(rows):

                for j in range(columns):
                    if j == 0:
                        txt = sep + "* - "
                    else:
                        txt = sep + "  - "

                    txt = txt + str(lstt[i][j]) + "\n"
                    f.write(txt)

        f.close()

    def parse_table_xls_to_list_table_read_xls(self, path_xls, sheet, _rows, _columns):

        # Give the location of the file
        if os.path.isfile(path_xls):
            loc = (path_xls)
        else:
            self.error("Path path_xls does not exit")

        # To open Workbook
        wb = xlrd.open_workbook(loc)
        if sheet is None:
            sheet = wb.sheet_by_index(0)
        else:
            sheet = wb.sheet_by_name(sheet)

        if sheet is None: self.error("Sheet name does not exit")

        # For row 0 and column 0
        sheet.cell_value(0, 0)

        (rows,columns) = (_rows,_columns)
        if _rows is None: rows = sheet.nrows
        if _columns is None: columns = sheet.ncols

        # populate list of lists,
        lst = []
        for i in range(rows):
            lst.append([])
            for j in range(columns):
                lst[i].append(sheet.cell_value(i, j))

        return (rows, columns, lst)

########################################################
########################################################
def test_parse_table_xls_to_list_table():

    import conf_lib as conf_lib
    clib = conf_lib.conf_lib()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0,current_directory)

    data_dir = os.path.join(current_directory, "_data")
    tables_dir = os.path.join(current_directory, "tables")
    images_dir = os.path.join(current_directory, "images")

    file_name = "Details_per_machine"
    output_path = os.path.join(tables_dir, file_name + ".rst")
    path_xls = os.path.join(data_dir, file_name + ".xls")
    clib.parse_table_xls_to_list_table(path_xls, output_path, caption = None, sheet = None, rows = None, columns = None)

def parse_figures_files_table():

    import conf_lib as conf_lib
    clib = conf_lib.conf_lib()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0,current_directory)

    data_dir = os.path.join(current_directory, "_data")
    tables_dir = os.path.join(current_directory, "tables")
    images_dir = os.path.join(current_directory, "images")

    file_name = "Images"
    path_xls = os.path.join(data_dir, file_name + ".xls")
    sheet = "parse_figures_files"
    clib.parse_figures_files_table(path_xls, sheet, images_dir)

def tests():
    #parse_table_xls_to_list_table
    #test_parse_table_xls_to_list_table()
    #parse_figures_files_table
    parse_figures_files_table()

if __name__ == "__main__":

    tests()
