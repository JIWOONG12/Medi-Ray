package com.example.MR.controller;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.interceptor.TransactionAspectSupport;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import com.example.MR.entity.ImgTb;
import com.example.MR.entity.ResultTb;
import com.example.MR.entity.UploadRequestDto;
import com.example.MR.mapper.DiagnosisMapper;
import com.example.MR.mapper.MRMapper;


@RestController
@RequestMapping("/api/upload")
public class DicomController {

	private final TransactionTemplate transactionTemplate;	
	
	@Autowired
    public DicomController(TransactionTemplate transactionTemplate) {
        this.transactionTemplate = transactionTemplate;
    }
	
    @Value("${fastapi.url}")
    private String fastapiUrl;

    @Autowired
    private MRMapper mapper;
    @Autowired
    private DiagnosisMapper dmapper;

    @PostMapping("/dicom")
    public ResponseEntity<?> uploadDicom(@RequestParam("file") MultipartFile file) {
        try {
            RestTemplate restTemplate = new RestTemplate();

            // FastAPI로 전송할 요청 데이터 설정
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            Resource dicomResource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };

            HttpEntity<Resource> requestEntity = new HttpEntity<>(dicomResource, headers);

            // FastAPI로 파일 전송
            ResponseEntity<Map> response = restTemplate.exchange(fastapiUrl + "/dupload", HttpMethod.POST, requestEntity, Map.class);

            return ResponseEntity.ok(response.getBody());  // FastAPI의 JSON 응답을 그대로 클라이언트에 반환

        } catch (IOException e) {
            return ResponseEntity.status(500).body("파일 처리 중 오류가 발생했습니다: " + e.getMessage());
        }
    }

    // 의사 정보 받아오기
    @GetMapping("/dtsearch")
    @ResponseBody
    public Map<String, String> diagnosis(@RequestParam("ptCode") String ptCode) {
        String dtName = mapper.DtName(ptCode);
        return Map.of("dtName", dtName);
    }
    

    @PostMapping("/imgupload")
    @Transactional(rollbackFor = {IOException.class, ArrayIndexOutOfBoundsException.class})
    public ResponseEntity<?> savePngAndResult(@RequestBody UploadRequestDto requestDto) throws Exception {
        try {
        	 
            
            // Base64 인코딩된 데이터에서 헤더 제거 및 디코딩
            String base64Data = requestDto.getImage();
            byte[] imageBytes = Base64.getDecoder().decode(base64Data);

            // 파일 경로 설정 및 저장
            String uploadDir = "C:\\Users\\USER\\dicom\\" + requestDto.getPtCode(); // 실제 경로로 변경
            File dir = new File(uploadDir);
            if (!dir.exists()) {
                dir.mkdirs(); // 디렉토리가 없으면 생성
            }

            String filePath = uploadDir + "\\" + requestDto.getFileName() + ".png";
            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                fos.write(imageBytes);
            }

            // `ImgTb` 객체 생성 및 데이터 설정
            String dtCode = mapper.DtCode(requestDto.getPtCode());

            ImgTb imgTb = ImgTb.builder()
                                .xrayCode(requestDto.getFileName())
                                .ptCode(requestDto.getPtCode())
                                .xrayImgPath(filePath) // 이미지 경로 설정
                                .dtCode(dtCode)
                                .build();
            	
            // 데이터베이스에 업로드  
            dmapper.imgUpload(imgTb);
            
            
            // 3. RESULT_TB 데이터 저장
            ResultTb resultTb = ResultTb.builder()
                    .xrayCode(requestDto.getFileName())
                    .ptCode(requestDto.getPtCode())
                    .Atelectasis(requestDto.getModelResult().get("Atelectasis"))
                    .Cardiomegaly(requestDto.getModelResult().get("Cardiomegaly"))
                    .Consolidation(requestDto.getModelResult().get("Consolidation"))
                    .Edema(requestDto.getModelResult().get("Edema"))
                    .Enlarged_Cardiomediastinum(requestDto.getModelResult().get("Enlarged Cardiomediastinum"))
                    .Fracture(requestDto.getModelResult().get("Fracture"))
                    .Lung_Lesion(requestDto.getModelResult().get("Lung Lesion"))
                    .Lung_Opacity(requestDto.getModelResult().get("Lung Opacity"))
                    .No_Finding(requestDto.getModelResult().get("No Finding"))
                    .Pleural_Effusion(requestDto.getModelResult().get("Pleural Effusion"))
                    .Pleural_Other(requestDto.getModelResult().get("Pleural Other"))
                    .Pneumonia(requestDto.getModelResult().get("Pneumonia"))
                    .Pneumothorax(requestDto.getModelResult().get("Pneumothorax"))
                    .Support_Devices(requestDto.getModelResult().get("Support Devices"))
                    .build();
            dmapper.imgReult(resultTb);

            return ResponseEntity.ok(Map.of("message", "이미지와 결과가 성공적으로 저장되었습니다."));

        } catch (IOException IOe) {
        	TransactionAspectSupport.currentTransactionStatus().setRollbackOnly();
        	return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("저장 중 오류 발생: " + IOe.getMessage());
        	
        } catch (ArrayIndexOutOfBoundsException AIOOBe) {
        	TransactionAspectSupport.currentTransactionStatus().setRollbackOnly();
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("데이터 처리 중 오류 발생: " + AIOOBe.getMessage());
        }
    }
}
