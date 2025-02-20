### Explain this two functions: 
```javascript
encodeBase64(string) {
    return btoa(unescape(encodeURIComponent(string)));
}

// Function to decode Base64 to original string
function decodeBase64(encodedStr) {
    const missingPadding = encodedStr.length % 4;
    if (missingPadding) {
        encodedStr += "=".repeat(4 - missingPadding);  // Add necessary padding
    }
    return decodeURIComponent(escape(atob(encodedStr)));
}
```
### Let's break down the two functions, `encodeBase64` and `decodeBase64`, one by one:

### 1. **`encodeBase64(string)`**:
```javascript
function encodeBase64(string) {
    return btoa(unescape(encodeURIComponent(string)));
}
```
This function is responsible for encoding a string into Base64.

#### Step-by-step explanation:

1. **`encodeURIComponent(string)`**:
   - This function encodes a string to ensure that special characters (like spaces, punctuation, etc.) are correctly handled.
   - It converts characters like `!`, `#`, or non-ASCII characters (such as `è` or `😊`) into their UTF-8 percent-encoded form.
   - Example: `"Hello World!"` becomes `"Hello%20World%21"`.

2. **`unescape(...)`**:
   - The `unescape()` function decodes percent-encoded characters. It's used here to "undo" any encoding made by `encodeURIComponent`.
   - In modern JavaScript, `unescape` is deprecated, and it's generally recommended to use `decodeURIComponent` instead, but in this context, it's used to help ensure that encoded characters are correctly handled for the next step.
   
3. **`btoa(...)`**:
   - This function is built into JavaScript and encodes a string into Base64. It only works on ASCII characters and requires that the input be a valid string in UTF-8 or ASCII.
   - Base64 encoding is used to convert binary data into an ASCII string, making it safe for transmission in URLs, headers, or other text-based environments.

**Overall**:
- The combination of `encodeURIComponent` and `unescape` ensures that the string is first encoded into a safe UTF-8 format and then Base64-encoded. This is necessary when the string might contain special or non-ASCII characters that need proper handling before the Base64 conversion.

---

### 2. **`decodeBase64(encodedStr)`**:
```javascript
function decodeBase64(encodedStr) {
    const missingPadding = encodedStr.length % 4;
    if (missingPadding) {
        encodedStr += "=".repeat(4 - missingPadding);  // Add necessary padding
    }
    return decodeURIComponent(escape(atob(encodedStr)));
}
```
This function decodes a Base64-encoded string back into its original format.

#### Step-by-step explanation:

1. **Padding Handling**:
   ```javascript
   const missingPadding = encodedStr.length % 4;
   if (missingPadding) {
       encodedStr += "=".repeat(4 - missingPadding);  // Add necessary padding
   }
   ```
   - Base64 strings need to have their lengths a multiple of 4. If the length of the Base64 string isn't a multiple of 4, the decoding may fail or be incorrect.
   - This code checks for any missing padding (`=`) by calculating `encodedStr.length % 4`.
   - If padding is missing, it adds the appropriate number of `=` characters to the end of the string to make its length a multiple of 4.

2. **`atob(encodedStr)`**:
   - `atob()` is a JavaScript function that decodes a Base64-encoded string into a plain ASCII string.
   - Base64 encoding represents binary data as a string of ASCII characters, so `atob()` reverses this encoding.
   - After decoding with `atob()`, you get a string that might contain special characters or non-ASCII characters, which need further decoding.

3. **`escape(...)`**:
   - The `escape()` function encodes special characters into a format that `decodeURIComponent` can understand. It replaces characters that aren't valid ASCII with their escape sequences.
   - **Note**: `escape()` is deprecated in modern JavaScript in favor of `encodeURIComponent`, but in this case, it was used to ensure that the non-ASCII characters are handled properly after Base64 decoding.

4. **`decodeURIComponent(...)`**:
   - This function decodes percent-encoded characters (like `%20` for spaces) into their original characters.
   - In this function, after we use `escape()` to handle special characters, `decodeURIComponent()` is used to get the original string back to its non-encoded form.

**Overall**:
- The function first handles any necessary padding, decodes the Base64 string with `atob()`, and then ensures that any special characters are decoded back to their original form using `escape()` and `decodeURIComponent()`.

---

### **Why are these functions used together?**

- **Encoding**: `encodeBase64` prepares strings for transmission by encoding special characters (like emojis or spaces) properly before Base64 encoding them.
- **Decoding**: `decodeBase64` reverses the encoding process, handling padding and decoding both the Base64 string and any percent-encoded characters back to their original format.

These functions are helpful when you need to safely store or transmit strings (especially those containing special characters or binary data) using Base64 encoding, and then later decode them back to their original form.
