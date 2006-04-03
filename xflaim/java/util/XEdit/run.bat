del xedit\*.class
copy c:\work\openflaim\trunk\xflaim\build\win-x86-32\debug\java\xflaimjni.jar
copy c:\work\openflaim\trunk\xflaim\build\win-x86-32\debug\lib\shared\xflaimjni.dll
"c:\Program Files\Java\jdk1.5.0_06\bin\javac.exe" -g -classpath xflaimjni.jar;. xedit\*.java
"c:\Program Files\Java\jdk1.5.0_06\bin\java.exe" -classpath xflaimjni.jar;. xedit.XEdit

:done
