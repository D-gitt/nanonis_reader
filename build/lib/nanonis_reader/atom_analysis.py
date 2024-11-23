# Drift correction of STM images obtained on monoclinic Ta2NiSe5.
def Ta2NiSe5_driftcorr(z, a_image_nm, c_image_nm, scansize_nm, origin_line=0, origin_pixel=0):
    import numpy as np
    lines, pixels = np.shape(z)
    a, c = 0.34916, 1.565 # lattice constants (nm)
    lines_corr = int(round(lines * (a_image_nm / a)))
    pixels_corr = int(round(pixels * (c_image_nm / c)))
    # print (lines_corr, pixels_corr)
    z_corr = np.copy(z)[origin_line:origin_line+lines_corr, origin_pixel:origin_pixel+pixels_corr]
    
    # length of scansize must be 2.
    # unitpx in \AA unit.
    a_unitpx = (10*scansize_nm[0])/lines_corr
    c_unitpx = (10*scansize_nm[1])/pixels_corr

    return z_corr, a_unitpx, c_unitpx


# Extract topograph from a ASCII XYZ file of WSxM.
def Extract_Z (Path, Search = 'X[nm]'):
    import numpy as np
    # Search = 'X[nm]' # X[nm]가 포함된 줄 찾기
    with open(Path, 'r', encoding = 'ISO-8859-1') as f: # encoding='ISO-8859-1'
        for line_number, line in enumerate(f):  # enumerate: 대상의 원소와 index를 묶은 tuple을 반환. / start=1 : index가 1부터 시작. (default==0)
            if Search in line:
                print(f"Word '{Search}' found on line {line_number+1}")
                break # line_number == 2
        XYZ_data = f.readlines() # line_number + 1 번째부터 한줄씩 모든 줄을 읽어들임.
        # print(XYZ_data) # 읽어들인 결과값 출력.

#     Extract z data and do 3sigma
        Result = []
        for Z in XYZ_data[1:]:
            Result.append(Z.split('\t')[2].strip('\n')) # z data만 추출
        Z_data = np.array([float(Z) for Z in Result]) # String array to float array in pm.
    print ('Finished.')
    return Z_data