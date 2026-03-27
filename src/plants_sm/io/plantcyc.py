def data_by_record(filename):
    """
    Auxiliar function to read the dat file and divide the info by record
    Parameters
    ----------
    filename: str
    json_path for the dat file

    Returns
    -------
    all_records: list
    list of dicts where each dict has the info of each record
    """
    all_records = []
    datafile = open(filename, 'r', encoding='ISO-8859-1')
    line = datafile.readline().strip()

    while line:
        if line[0] != "#" and "UNIQUE-ID" in line:
            record = []
            while line != '//':
                record.append(line)
                line = datafile.readline().strip()

            record_dic = {}
            for info in record:
                if info.count(' - ') == 1:
                    key, value = info.split(' - ')
                    if key not in record_dic:
                        record_dic[key] = [value]
                    else:
                        record_dic[key].append(value)

            all_records.append(record_dic)
        else:
            line = datafile.readline().strip()

    return all_records