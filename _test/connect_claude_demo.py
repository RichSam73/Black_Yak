
import os
import anthropic

# 1. API 키 설정
# 환경 변수에 'ANTHROPIC_API_KEY'가 설정되어 있거나, 아래에 직접 입력해야 합니다.
# 주의: 실제 프로젝트에서는 API 키를 코드에 직접 노출하지 마세요. (환경 변수 또는 .env 파일 권장)
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    print("⚠️ API 키가 설정되지 않았습니다.")
    print("방법 1: 터미널에서 `set ANTHROPIC_API_KEY=sk-...` (Windows) 또는 `export ANTHROPIC_API_KEY=sk-...` (Mac/Linux) 실행")
    print("방법 2: 이 스크립트의 api_key 변수에 직접 키 문자열 할당 (보안 주의)")
    # api_key = "sk-..." 
    exit(1)

# 2. 클라이언트 초기화
client = anthropic.Anthropic(
    api_key=api_key,
)

print("🚀 Claude에게 인사를 요청합니다...")

try:
    # 3. 메시지 생성 요청
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "안녕, Claude! 간단한 자기소개를 해줘."}
        ]
    )

    # 4. 응답 출력
    print("\n[Claude의 응답]")
    print(message.content[0].text)

except anthropic.APIConnectionError as e:
    print("🔥 서버 연결 오류:", e)
    print("인터넷 연결을 확인하세요.")
except anthropic.AuthenticationError as e:
    print("🔒 인증 오류:", e)
    print("API 키가 올바른지 확인하세요.")
except Exception as e:
    print("❌ 오류 발생:", e)
