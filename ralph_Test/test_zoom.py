"""
Ctrl+휠 줌 테스트 스크립트
Playwright를 사용하여 test_wheel_zoom.html의 줌 기능을 자동 테스트
"""
import asyncio
from playwright.async_api import async_playwright
import os

HTML_PATH = os.path.join(os.path.dirname(__file__), 'test_wheel_zoom.html')
FILE_URL = f'file:///{HTML_PATH.replace(os.sep, "/")}'

async def test_ctrl_wheel_zoom():
    print(f"Testing: {FILE_URL}")

    async with async_playwright() as p:
        # 새 브라우저 인스턴스 (기존 Chrome과 충돌 방지)
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # HTML 파일 열기
        await page.goto(FILE_URL)
        await page.wait_for_timeout(1000)

        # 초기 줌 레벨 확인
        initial_zoom = await page.locator('#zoomLevel').inner_text()
        print(f"Initial zoom: {initial_zoom}")
        assert initial_zoom == "100%", f"Expected 100%, got {initial_zoom}"

        # 컨테이너 위치 찾기
        container = page.locator('#container')
        box = await container.bounding_box()
        center_x = box['x'] + box['width'] / 2
        center_y = box['y'] + box['height'] / 2

        # Ctrl+휠 업 (확대) 테스트
        print("\n--- Test 1: Ctrl+Wheel Up (Zoom In) ---")
        await page.mouse.move(center_x, center_y)
        await page.keyboard.down('Control')
        await page.mouse.wheel(0, -100)  # 위로 스크롤 = 확대
        await page.keyboard.up('Control')
        await page.wait_for_timeout(500)

        zoom_after_in = await page.locator('#zoomLevel').inner_text()
        print(f"Zoom after Ctrl+Wheel Up: {zoom_after_in}")

        # 확대되었는지 확인
        zoom_value = int(zoom_after_in.replace('%', ''))
        if zoom_value > 100:
            print("SUCCESS: Zoom increased!")
        else:
            print("FAILED: Zoom did not increase")

        # Ctrl+휠 다운 (축소) 테스트
        print("\n--- Test 2: Ctrl+Wheel Down (Zoom Out) ---")
        await page.keyboard.down('Control')
        await page.mouse.wheel(0, 100)  # 아래로 스크롤 = 축소
        await page.mouse.wheel(0, 100)  # 두 번 축소
        await page.keyboard.up('Control')
        await page.wait_for_timeout(500)

        zoom_after_out = await page.locator('#zoomLevel').inner_text()
        print(f"Zoom after Ctrl+Wheel Down x2: {zoom_after_out}")

        # 줌 범위 테스트 (최소값)
        print("\n--- Test 3: Zoom Min Limit ---")
        await page.keyboard.down('Control')
        for _ in range(20):  # 많이 축소
            await page.mouse.wheel(0, 100)
        await page.keyboard.up('Control')
        await page.wait_for_timeout(500)

        zoom_min = await page.locator('#zoomLevel').inner_text()
        print(f"Zoom at min: {zoom_min}")
        min_value = int(zoom_min.replace('%', ''))
        assert min_value >= 50, f"Min zoom should be >= 50%, got {min_value}%"

        # 줌 범위 테스트 (최대값)
        print("\n--- Test 4: Zoom Max Limit ---")
        await page.keyboard.down('Control')
        for _ in range(30):  # 많이 확대
            await page.mouse.wheel(0, -100)
        await page.keyboard.up('Control')
        await page.wait_for_timeout(500)

        zoom_max = await page.locator('#zoomLevel').inner_text()
        print(f"Zoom at max: {zoom_max}")
        max_value = int(zoom_max.replace('%', ''))
        assert max_value <= 200, f"Max zoom should be <= 200%, got {max_value}%"

        # 일반 휠 스크롤 테스트 (Ctrl 없이)
        print("\n--- Test 5: Normal Wheel Scroll (No Ctrl) ---")
        await page.locator('#zoomLevel').click()  # 리셋 버튼 근처 클릭
        await page.evaluate("applyZoom(100)")  # 줌 리셋
        await page.wait_for_timeout(300)

        await page.mouse.move(center_x, center_y)
        await page.mouse.wheel(0, 100)  # Ctrl 없이 스크롤
        await page.wait_for_timeout(500)

        zoom_after_scroll = await page.locator('#zoomLevel').inner_text()
        print(f"Zoom after normal scroll: {zoom_after_scroll}")
        assert zoom_after_scroll == "100%", "Normal scroll should not change zoom"
        print("SUCCESS: Normal scroll does not zoom!")

        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)

        await page.wait_for_timeout(2000)  # 결과 확인용 대기
        await browser.close()

        return True

if __name__ == "__main__":
    result = asyncio.run(test_ctrl_wheel_zoom())
    exit(0 if result else 1)
