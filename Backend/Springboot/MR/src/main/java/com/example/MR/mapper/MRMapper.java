package com.example.MR.mapper;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import com.example.MR.entity.DoctorTb;
import com.example.MR.entity.PatientTb;

//@Mapper
public interface MRMapper {
	// 회원가입
	// 회원가입 정보 입력하기
	@Insert("INSERT INTO DOCTOR_TB(DT_CODE, DT_ID, DT_PW, DT_NAME, DIVISION, DT_TELNO) "
			+ "VALUES(#{DT_CODE}, #{DT_PW}, #{DT_PW}, #{DT_NAME}, #{DIVISION}, #{DT_TELNO})")
	public void registDoctor(DoctorTb dt);

	// 로그인, 의사테이블에서 중복된 id가 있는지 확인하기, 마이페이지
	@Select("SELECT * FROM DOCTOR_TB WHERE BINARY DT_ID = #{DT_ID}")
	public DoctorTb checkDuplicateId(String DT_ID);

	// 의사테이블에서 code 만 빼오기
	@Select("SELECT * FROM DOCTOR_TB WHERE BINARY DT_CODE = #{DT_CODE}")
	public DoctorTb checkDuplicateCode(String DT_CODE);

	/* Security 적용하면서 pw는 알아서 비교해줌
	 * // 로그인 // 로그인 시 일치하는 지(넘어오는 값이 있는지로 판단) // 로그인 후에 반환되는 정보에 이름만 따로 띄우기도 해야함
	 * java에서 빼오기
	 * 
	 * @Select("SELECT * FROM DOCTOR_TB " +
	 * "WHERE BINARY DT_ID = #{dtId} AND BINARY DT_PW = #{dtPw}") public DoctorTb
	 * validateLogin(String dtId, String dtPw);
	 */

//	// 마이페이지 // 여기서 이름만 추출해서 로그인 후에 이름 뜨게 하면 될 듯...?
//	@Select("SELECT * FROM DOCTOR_TB WHERE DT_ID = #{DT_ID}")
//	public ArrayList<DoctorTb> dtSelect(String DT_ID);

	// 마이페이지입력값 수정(비밀번호 변경가능)
	@Update("UPDATE DOCTOR_TB SET DT_PW = #{DT_PW} " 
			+ "WHERE DT_ID = #{DT_ID}")
	public void updateDoctorPw(String DT_ID, String DT_PW);

	// 마이페이지입력값 수정(의사소속 변경가능)
	@Update("UPDATE DOCTOR_TB SET DIVISION = #{DIVISION} " 
			+ "WHERE DT_ID = #{DT_ID}")
	public void updateDivision(String DT_ID, String DIVISION);

	// 업로드 페이지
	// 같은 환자 번호를 가지고 있는 의사의 이름 불러오기
	@Select("SELECT d.DT_NAME AS dtName FROM DOCTOR_TB d " 
			+ "JOIN PATIENT_TB p ON p.DT_CODE = d.DT_CODE  "
			+ "WHERE p.PT_CODE = #{PT_CODE} ")
	public String DtName(String PT_CODE);

	// 환자에 대한 의사코드 매칭
	@Select("SELECT DT_CODE FROM PATIENT_TB WHERE PT_CODE = #{PT_CODE}")
	public String DtCode(String PT_CODE);

	// 환자 정보 업로드
//		@Insert("INSERT INTO PATIENT_TB(PT_CODE, PT_NAME, PT_BIRTHDATE, PT_GEN, DT_CODE")

	// 진단(환자검색)
	// 입력한 환자에 맞는 리스트 반환, img테이블에 있는 날짜순대로 내림차순에서 반환할 수 있도록 하기
	@Select("SELECT p.*, d.DT_NAME AS DT_NAME, i.XRAY_CODE " 
			+ "FROM PATIENT_TB p "
			+ "JOIN IMG_TB i ON p.PT_CODE = i.PT_CODE " 
			+ "JOIN DOCTOR_TB d ON p.DT_CODE = d.DT_CODE "
			+ "WHERE (p.PT_NAME LIKE CONCAT('%', #{ptName}, '%') OR p.PT_CODE = #{ptCode}) "
			+ "AND i.XRAY_DATE = ( " 
			+ "    SELECT MAX(i2.XRAY_DATE) " 
			+ "    FROM IMG_TB i2 "
			+ "    WHERE i2.PT_CODE = i.PT_CODE " 
			+ ") " 
			+ "ORDER BY i.XRAY_DATE DESC")
	public List<PatientTb> patientList(String inputValue);

	// 불필요한 연산을 뺀다... 성능 고려.. 자바에서 처리해 알간..?

}
