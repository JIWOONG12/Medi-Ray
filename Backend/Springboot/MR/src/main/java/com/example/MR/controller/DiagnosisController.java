package com.example.MR.controller;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import com.example.MR.entity.ImgTb;
import com.example.MR.entity.LabelProbability;
import com.example.MR.entity.PatientTb;
import com.example.MR.entity.ResultTb;
import com.example.MR.mapper.DiagnosisMapper;
import com.example.MR.mapper.MRMapper;

@RestController
@RequestMapping("/diagnosis")
public class DiagnosisController {
	
	@Value("${fastapi.url}")
    private String fastapiUrl;

    @Autowired
    private MRMapper mapper;
    @Autowired
    private DiagnosisMapper dmapper;

    // 환자 검색
    @GetMapping("/search")
    @ResponseBody
    public List<PatientTb> diagnosis(@RequestParam("inputValue") String inputValue) {
        return mapper.patientList(inputValue); // 여러 결과를 반환
    }

    // 기존 의사소견 조회
    @GetMapping("/xray/dtOpinion")
    public ResponseEntity<String> getOpinion(@RequestParam String xrayCode) {
        try {
            String existingOpinion = dmapper.checkOpinion(xrayCode);
            if (existingOpinion != null) {
                return ResponseEntity.ok(existingOpinion);
            } else {
                return ResponseEntity.ok("");
            }
        } catch (Exception e) {
            e.printStackTrace(); 
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("서버 오류가 발생했습니다.");
        }
    }

    // 의사소견 업데이트
    @PostMapping("/xray/doUpload")
    public ResponseEntity<String> UpdateOpinion(@RequestParam String xrayCode, @RequestParam String dtOpinion) {
        if (dmapper.updateOpinion(xrayCode, dtOpinion) > 0) {
            return ResponseEntity.ok("의견이 업데이트되었습니다.");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("업데이트할 의견이 존재하지 않습니다.");
        }
    }
    
    // 촬영리스트 불러오기
    @GetMapping("/xray/dateList")
    public ArrayList<ImgTb> dateList(@RequestParam("ptCode") String ptCode) {
        return dmapper.imgList(ptCode);
    }
    
    // 촬영리스트 클릭하면 해당 페이지로 이동하기
    @GetMapping("/xray/imgDate")
    public ArrayList<ImgTb> imgDate(@RequestParam String ptCode, @RequestParam String xrayDate){
        return dmapper.imgDate(ptCode, xrayDate);
    }

    // 이미지 가져오기
    private static final String UPLOAD_DIR = "C:/Users/USER/dicom"; 

    @GetMapping("/xray/getImage")
    public ResponseEntity<Resource> getImage(@RequestParam("ptCode") String ptCode, @RequestParam("xrayCode") String xrayCode) {
        try {
            String filePath = UPLOAD_DIR + "/" + ptCode + "/" +xrayCode + ".png";
            Resource resource = new FileSystemResource(filePath);

            if (!resource.exists()) {
            	return ResponseEntity.status(HttpStatus.NOT_FOUND).build(); // 파일이 없을 때 404 반환
            }

            HttpHeaders headers = new HttpHeaders();
            headers.add(HttpHeaders.CONTENT_TYPE, "image/png");
            return new ResponseEntity<>(resource, headers, HttpStatus.OK);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    // 저장된 결과 중에서 50%를 넘고 상위 4개인 것만 출력하기
    @GetMapping("/xray/result")
    public List<LabelProbability> getTop4Probs(@RequestParam("xrayCode") String xrayCode) throws IllegalAccessException {
        // 1. 데이터베이스에서 xrayCode에 해당하는 결과 리스트를 가져옴
        ArrayList<ResultTb> results = dmapper.resultList(xrayCode);
        
        if (results.isEmpty()) {
            System.out.println("No data found for xrayCode: " + xrayCode);
            return Collections.emptyList(); // 데이터가 없을 경우 빈 리스트 반환
        }

        // 2. 상위 4개의 확률과 라벨명을 담을 리스트 생성
        List<LabelProbability> labelProbabilities = new ArrayList<>();

        // 3. ResultTb 객체에서 각 필드를 순회하여 확률 값을 추출
        for (ResultTb result : results) {
            // ResultTb 클래스의 모든 필드를 가져옴
            Field[] fields = ResultTb.class.getDeclaredFields();

            for (Field field : fields) {
                // 필드가 double 또는 Double 타입인지 확인
                if (field.getType() == double.class || field.getType() == Double.class) {
                    field.setAccessible(true); // private 필드 접근 허용
                    Double value = (Double) field.get(result); // 필드의 값 가져오기

                    // 50.0 이상인 확률 값만 리스트에 추가
                    if (value != null && value > 50.0) {
                    	// 필드명에서 "_"를 " "로 변환하여 라벨명을 생성
                        String labelName = field.getName().replace("_", " ");
                        // 필드명과 확률 값을 LabelProbability 객체에 담아 리스트에 추가
                        labelProbabilities.add(new LabelProbability(labelName, value));
                    }
                }
            }
        }
        

        // 4. 리스트를 확률 값이 큰 순서대로 정렬하고 상위 4개만 남겨 반환
        return labelProbabilities.stream()
               .sorted(Comparator.comparing(LabelProbability::getProbability).reversed()) // 확률 값을 기준으로 내림차순 정렬
               .limit(4) // 상위 4개의 항목만 선택
               .collect(Collectors.toList()); // 결과를 List<LabelProbability> 형태로 반환
    }


    
    // Grad-CAM 생성 요청 처리
	// Grad-CAM 생성 요청 처리
    @PostMapping("/gradcam")
    public ResponseEntity<?> generateGradCam(@RequestBody Map<String, String> requestBody) {
        try {
            RestTemplate restTemplate = new RestTemplate(); // REST API 호출을 위한 RestTemplate 생성

            // FastAPI로 요청 전송을 위한 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON); // JSON 형식으로 설정

            // 요청 본문에서 Base64 이미지 데이터 추출
            String base64Img = requestBody.get("base64_img");
            if (base64Img == null || base64Img.isEmpty()) {
                return ResponseEntity.badRequest().body("Base64 이미지 데이터가 없습니다.");
            }

            // FastAPI로 보낼 JSON 데이터 구성
            Map<String, String> fastApiRequestBody = new HashMap<>();
            fastApiRequestBody.put("base64_img", base64Img);

            HttpEntity<Map<String, String>> requestEntity = new HttpEntity<>(fastApiRequestBody, headers);

            // FastAPI로 POST 요청 전송
            ResponseEntity<Map> response = restTemplate.exchange(
                fastapiUrl + "/gradcam",  // FastAPI의 Grad-CAM 엔드포인트
                HttpMethod.POST,
                requestEntity,
                Map.class // 응답 데이터를 Map 형태로 받음
            );

            // FastAPI의 응답을 클라이언트로 반환
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            // 예외 발생 시 에러 메시지 반환
            return ResponseEntity.status(500).body("Grad-CAM 생성 중 오류가 발생했습니다: " + e.getMessage());
        }
    }
    
    
}
