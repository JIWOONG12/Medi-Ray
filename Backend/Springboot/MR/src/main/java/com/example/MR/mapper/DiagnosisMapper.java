package com.example.MR.mapper;

import java.util.ArrayList;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import com.example.MR.entity.ImgTb;
import com.example.MR.entity.ResultTb;

@Mapper
public interface DiagnosisMapper {
	// 업로드 페이지(이미지 업로드)
	@Insert("INSERT INTO IMG_TB(XRAY_CODE, PT_CODE, XRAY_DATE, XRAY_IMG_PATH, DT_CODE) "
			+ "VALUES (#{xrayCode}, #{ptCode}, Current_timestamp, #{xrayImgPath}, #{dtCode})")
	public int imgUpload(ImgTb it);
	// 모델 진단 결과도 같이 업로드
	@Insert("INSERT INTO RESULT_TB(XRAY_CODE, PT_CODE, ATELECTASIS_PROB, CARDIOMEGALY_PROB, CONSOLIDATION_PROB, EDEMA_PROB, ENLARGED_CARDIO_PROB, FRACTURE_PROB, LUNG_LESION_PROB, LUNG_OPACITY_PROB, NO_FIND_PROB, PLEURAL_EFFUSION_PROB, PLEURAL_OTHER_PROB, PNEUMONIA_PROB, PNEUMOTHORAX_PROB, OTHER_PROB) "
			+ "VALUES (#{xrayCode}, #{ptCode}, #{Atelectasis}, #{Cardiomegaly}, #{Consolidation}, #{Edema}, #{Enlarged_Cardiomediastinum}, #{Fracture}, #{Lung_Lesion}, #{Lung_Opacity}, #{No_Finding}, #{Pleural_Effusion}, #{Pleural_Other}, #{Pneumonia}, #{Pneumothorax}, #{Support_Devices})")
	public int imgReult(ResultTb rt);

	// 의사진단입력란(값이 존재하는지 확인)
	@Select("SELECT DT_OPINION FROM IMG_TB WHERE XRAY_CODE = #{xrayCode}")
	public String checkOpinion(String xrayCode);
		
	// 값이 존재한다면
	@Update("UPDATE IMG_TB SET DT_OPINION = #{dtOpinion} WHERE XRAY_CODE = #{xrayCode}")
	public int updateOpinion(@Param("xrayCode") String xrayCode, @Param("dtOpinion") String dtOpinion);

	// 검색해서 들어가면 다이콤 영상까지 포함해서 들어가게
	// 촬영기록
	@Select("SELECT * FROM IMG_TB " 
			+ "WHERE PT_CODE = #{ptCode} "
			+ "ORDER BY XRAY_DATE DESC")
	public ArrayList<ImgTb> imgList(String ptCode);
	
	// 촬영기록에 있는 날짜 클릭하면 해당 값이 나오도록 지정, 이미지도 끌어오기
	@Select("SELECT * FROM IMG_TB WHERE (XRAY_DATE = #{xrayDate} AND PT_CODE = #{ptCode})")
	public ArrayList<ImgTb> imgDate(@Param("ptCode") String ptCode, @Param("xrayDate") String xrayDate);

	// 다이콤에 대한 결과 반환
	@Select("SELECT * FROM RESULT_TB "
			+ "WHERE XRAY_CODE = #{xrayCode}") 
	public ArrayList<ResultTb> resultList(String xrayCode);

}
