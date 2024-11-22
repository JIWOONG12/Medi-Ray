package com.example.MR.config;

import org.springframework.beans.factory.annotation.Autowired;

// 이 클래스는 Spring Security 설정을 정의하는 파일입니다.

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;

import com.example.MR.service.CustomUserDetailsService;

import jakarta.servlet.http.HttpServletResponse;

@Configuration
// Spring에서 이 클래스를 설정 파일로 인식하도록 @Configuration 어노테이션을 추가합니다.
public class SecurityConfig {
	
	@Autowired
	private CustomUserDetailsService customUserDetailsService;
	
	
	@Bean
	// Spring Security의 필터 체인을 구성하는 Bean을 등록합니다.
	public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
		http.csrf(csrf -> csrf.disable()) // CSRF 보호를 비활성화합니다 (필요시 활성화 가능).
				.authorizeHttpRequests(auth -> auth
						.requestMatchers("/index", "/login/**", "/signup/**", "/static/**", "/assets/**").permitAll() // /인증
						.anyRequest().authenticated() // 나머지 모든 요청은 인증이 필요합니다.
				).formLogin(login -> login // 폼 로그인 기능을 활성화합니다.
						.loginPage("/login") // 로그인 페이지의 경로를 "/login"으로 지정합니다.
						.loginProcessingUrl("/login/process") // 로그인 요청 처리 경로
						.defaultSuccessUrl("/main", true) // 로그인 성공 시 리다이렉트할 경로
						.failureHandler(customFailureHandler()) // 로그인 실패 시 리다이렉트할 경로
						.permitAll() // 로그인 페이지는 모든 사용자(인증되지 않은 사용자 포함)가 접근할 수 있습니다.
				).logout(logout -> logout // 로그아웃 기능을 설정합니다.
						.logoutUrl("/logout") // 로그아웃 요청 경로를 "/logout"으로 지정합니다.
						.logoutSuccessUrl("/login?logout=true") // 로그아웃 성공 후 리다이렉트할 경로
				);

		return http.build(); // 최종적으로 설정을 완료하여 SecurityFilterChain 객체를 반환합니다.
	}
	
	@Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
        authProvider.setUserDetailsService(customUserDetailsService); // CustomUserDetailsService 설정
        authProvider.setPasswordEncoder(passwordEncoder()); // PasswordEncoder 설정
        return authProvider;
    }

	// 실패 시 JSON 응답을 보내기 위한 커스텀 핸들러
	@Bean
	public AuthenticationFailureHandler customFailureHandler() {
		return (request, response, exception) -> {
			System.out.println("로그인 실패: " + exception.getMessage()); // 실패 이유 로그
			response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
			response.setContentType("application/json;charset=UTF-8");
			response.getWriter().write("{\"error\": \"true\", \"message\": \"유효한 아이디나 비밀번호가 아닙니다.\"}");
		};
	}

	@Bean
	public PasswordEncoder passwordEncoder() {
		return new BCryptPasswordEncoder();
	}
	
//	@Bean
//	public UserDetailsService userDetailsService() {
//		// 예시: 인메모리 사용자 인증 정보 설정
//		UserDetails user = User.withUsername(user).password(passwordEncoder().encode("password")) // 비밀번호를 암호화하지 않고
//																									// {noop} 사용
////                               .roles("USER") // 사용자에게 "USER" 역할 부여
//				.build();
////		UserDetails user = detailService.loadUserByUsername(null);
//		return new InMemoryUserDetailsManager(user); // 메모리에서 사용자 인증 정보를 관리
//	}

	@Bean
	public UserDetailsService userDetailsService() {
	    return customUserDetailsService; // CustomUserDetailsService를 빈으로 등록
	}
	
	

	
	
}
