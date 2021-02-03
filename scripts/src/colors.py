

def _getColor(colorName):
    if (colorName == 'FG_DEFAULT'):return '39'
    elif (colorName == 'BG_DEFAULT'): return '49'
    elif (colorName == 'FG_BLACK'): return '30'
    elif (colorName == 'BG_BLACK'): return '40'
    elif (colorName == 'FG_RED'): return '31'
    elif (colorName == 'BG_RED'): return '41'
    elif (colorName == 'FG_GREEN'): return '32'
    elif (colorName == 'BG_GREEN'): return '42'
    elif (colorName == 'FG_YELLOW'): return '33'
    elif (colorName == 'BG_YELLOW'): return '43'
    elif (colorName == 'FG_BLUE'): return '34'
    elif (colorName == 'BG_BLUE'): return '44'
    elif (colorName == 'FG_MAGENTA'): return '35'
    elif (colorName == 'BG_MAGENTA'): return '45'
    elif (colorName == 'FG_CYAN'): return '36'
    elif (colorName == 'BG_CYAN'): return '46'
    elif (colorName == 'FG_WHITE'): return '37'
    elif (colorName == 'BG_WHITE'): return '47'
    elif (colorName == 'FG_BLACK_BRIGHT'): return '90'
    elif (colorName == 'BG_BLACK_BRIGHT'): return '100'
    elif (colorName == 'FG_RED_BRIGHT'): return '91'
    elif (colorName == 'BG_RED_BRIGHT'): return '101'
    elif (colorName == 'FG_GREEN_BRIGHT'): return '92'
    elif (colorName == 'BG_GREEN_BRIGHT'): return '102'
    elif (colorName == 'FG_YELLOW_BRIGHT'): return '93'
    elif (colorName == 'BG_YELLOW_BRIGHT'): return '103'
    elif (colorName == 'FG_BLUE_BRIGHT'): return '94'
    elif (colorName == 'BG_BLUE_BRIGHT'): return '104'
    elif (colorName == 'FG_MAGENTA_BRIGHT'): return '95'
    elif (colorName == 'BG_MAGENTA_BRIGHT'): return '105'
    elif (colorName == 'FG_CYAN_BRIGHT'): return '96'
    elif (colorName == 'BG_CYAN_BRIGHT'): return '106'
    elif (colorName == 'FG_WHITE_BRIGHT'): return '97'
    elif (colorName == 'BG_WHITE_BRIGHT'): return '107'
    elif (colorName == 'BOLD'): return '1'
    elif (colorName == 'FAINT'): return '2'
    elif (colorName == 'UNDERLINE'): return '4'
    elif (colorName == 'DOUBLE_UNDERLINE'): return '21'
    elif (colorName == 'BLINK'): return '5'
    else: return '39'


def print_col(msg, colorName):
    print('\033[1;' + _getColor(colorName) + 'm' + msg + '\033[0m')


if __name__ == "__main__":
    print_col("prova", 'FG_RED')