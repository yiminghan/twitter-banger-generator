import fs from "fs";

// @ts-check
const { test, expect } = require("@playwright/test");

// Use a throwaway account for this
const userName = "MeSocam48073";
const password = "YOUR PASSWORD HERE";

const scrapeFilePath = "./scrap";
const scrollSize = 2000;
const timeoutPerscroll = () => {
  // add some jitter per scroll so we look like an actuall human
  return 5000 + Math.random() * 1000;
};

// Your Favorite Twitter shitposter
const loopFor = 300;
//@jjohnpotter

test("Twitter TimeLine Scraper", async ({ page }) => {
  const twitterAt = "TheWeebDev";

  // upper bound of our execution time
  test.setTimeout(loopFor * 6000 + 10000);
  await page.setViewportSize({
    width: 1000,
    height: scrollSize,
  });

  await page.goto("https://x.com/");
  await page.click("text=Sign in");
  // Wait for page to load
  await page.waitForTimeout(5000);
  const userInput = "input";
  await page.fill(userInput, userName);
  await page.click("text=Next");
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `1.png` });
  await page.keyboard.type(password, { delay: 50 });
  await page.click("text=Log in");
  await page.waitForTimeout(2000);

  // We are logged into twitter now
  await page.goto("https://twitter.com/" + twitterAt);
  await page.waitForTimeout(4000);

  let tweets = [];
  // Start scrolling
  for (let i = 0; i < loopFor; i++) {
    await page.mouse.wheel(0, scrollSize);
    await page.waitForTimeout(timeoutPerscroll());
    tweets = tweets.concat(await page.getByTestId("tweetText").allInnerTexts());
    // Write immediately
    fs.writeFileSync(
      scrapeFilePath + "/" + twitterAt + ".txt",
      tweets.join("\n------\n"),
      "utf-8"
    );
  }
});

// test("Twitter Screenshot 2", async ({ page }) => {
//   const twitterAt = "realGeorgeHotz";

//   // upper bound of our execution time
//   test.setTimeout(loopFor * 6000 + 10000);
//   await page.setViewportSize({
//     width: 1000,
//     height: scrollSize,
//   });

//   await page.goto("https://x.com/");
//   await page.click("text=Sign in");
//   // Wait for page to load
//   await page.waitForTimeout(5000);
//   const userInput = "input";
//   await page.fill(userInput, userName);
//   await page.click("text=Next");
//   await page.waitForTimeout(2000);
//   await page.screenshot({ path: `1.png` });
//   await page.keyboard.type(password, { delay: 50 });
//   await page.click("text=Log in");
//   await page.waitForTimeout(2000);

//   // We are logged into twitter now
//   await page.goto("https://twitter.com/" + twitterAt);
//   await page.waitForTimeout(4000);

//   let tweets = [];
//   // Start scrolling
//   for (let i = 0; i < loopFor; i++) {
//     await page.mouse.wheel(0, scrollSize);
//     await page.waitForTimeout(timeoutPerscroll());
//     tweets = tweets.concat(await page.getByTestId("tweetText").allInnerTexts());
//     // Write immediately
//     fs.writeFileSync(
//       scrapeFilePath + "/" + twitterAt + ".txt",
//       tweets.join("\n------\n"),
//       "utf-8"
//     );
//   }
// });