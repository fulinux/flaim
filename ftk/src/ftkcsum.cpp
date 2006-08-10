//------------------------------------------------------------------------------
// Desc:	This file contains routines which calculates checksums
//
// Tabs:	3
//
//		Copyright (c) 1999-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: checksum.cpp 3123 2006-01-24 17:19:50 -0700 (Tue, 24 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#include "ftksys.h"

static FLMUINT32 *				gv_pui32CRCTbl = NULL;
static FLMBOOL						gv_bHaveFastCheckSum = FALSE;

#if defined( FLM_X86) && (defined( FLM_GNUC) || defined( FLM_WIN) || defined( FLM_NLM))

	#if defined( FLM_WATCOM_NLM)
	
		extern void ftkFastCheckSumMMX(
				void *				pBlk,
				unsigned long *	puiChecksum,	
				unsigned long *	puiXOR,
				unsigned long		uiNumberOfBytes);
		
		extern unsigned long ftkGetMMXSupported(void);
		
	#else
	
		static void ftkFastCheckSumMMX(
				void *				pBlk,
				unsigned long *	puiChecksum,	
				unsigned long *	puiXOR,
				unsigned long		uiNumberOfBytes);
		
		static unsigned long ftkGetMMXSupported(void);
		
	#endif

#elif defined( FLM_SPARC_PLUS)
	
		extern "C" void sparc_calc_checksum(
			void *				pBlk,
			unsigned long *	puiChecksum,	
			unsigned long *	puiXOR,
			unsigned long		uiNumberOfBytes);
		
		extern "C" void sparc_calc_xor(
			void *				pBlk,
			unsigned long *	puiXOR,
			unsigned long		uiNumberOfBytes);

#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_WATCOM_NLM) && defined( FLM_RING_ZERO_NLM)
	#pragma aux ftkGetMMXSupported parm;
	#pragma aux ftkGetMMXSupported = \
		0xB8 0x01 0x00 0x00 0x00            /* mov		eax, 1  				*/\
		0x0F 0xA2                         	/* CPUID  							*/\
		0x33 0xC0                         	/* xor		eax, eax 			*/\
		0xF7 0xC2 0x00 0x00 0x80 0x00       /* test		edx, (1 SHL 23) 	*/\
		0x0F 0x95 0xC0                      /* setnz	al  						*/\
		modify exact [EAX EBX ECX EDX];
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_WATCOM_NLM) && defined( FLM_LIBC_NLM)
unsigned long ftkGetMMXSupported( void)
{
	return( 1);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_WIN)
unsigned long ftkGetMMXSupported( void)
{
	unsigned long bMMXSupported;
	__asm
	{
		mov		eax, 1
		cpuid
		xor		eax, eax
		test		edx, (1 SHL 23)
		setnz		al
		mov		bMMXSupported, eax
	}
	
	return( bMMXSupported);
}
#endif
	
/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_32BIT) && defined( FLM_GNUC)
unsigned long ftkGetMMXSupported( void)
{
	FLMUINT32 	bMMXSupported;
	
	__asm__ __volatile__(
		"push		%%ebx\n"
		"mov		$1, %%eax\n"
		"cpuid\n"
		"xor		%%eax, %%eax\n"
		"test		$0x800000, %%edx\n"
		"setnz	%%al\n"
		"mov		%%eax, %0\n"
		"pop		%%ebx\n"
			: "=&r" (bMMXSupported)
			:
			: "%eax", "%ecx", "%edx");
	
	return( bMMXSupported);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_64BIT)
unsigned long ftkGetMMXSupported( void)
{
	return( 1);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_WATCOM_NLM)

	#pragma aux ftkFastCheckSumMMX parm [ESI] [eax] [ebx] [ecx];
	#pragma aux ftkFastCheckSumMMX = \
		0x50                          /* push	eax			;save the sum pointer  								*/\
		0x53                          /* push	ebx			;save the xor pointer 								*/\
		0x8B 0x10                     /* mov	edx, [eax]	;for local add 										*/\
		0x81 0xE2 0xFF 0x00 0x00 0x00 /* and	edx, 0ffh	;clear unneeded bits									*/\
		0x8B 0x1B                     /* mov	ebx, [ebx]	;for local xor 										*/\
		0x81 0xE3 0xFF 0x00 0x00 0x00 /* and	ebx, 0ffh	;clear unneeded bits									*/\
		0x8B 0xF9                     /* mov	edi, ecx		;save the amount to copy							*/\
		0x83 0xF9 0x20                /* cmp	ecx, 32		;see if we have enough for the big loop		*/\
		0x0F 0x82 0x63 0x00 0x00 0x00	/* jb		#MediumStuff 															*/\
		0xC1 0xE9 0x05                /* shr	ecx, 5		;convert length to 32 byte blocks				*/\
		0x83 0xE7 0x1F						/* and	edi, 01fh	;change saved length to remainder				*/\
		0x0F 0xEF 0xED						/* pxor	mm5, mm5		;wasted space to 16 byte align the loop		*/\
		0x0F 0x6E 0xE2						/* movd	mm4,edx																	*/\
		0x0F 0x6E 0xEB						/* movd	mm5,ebx																	*/\
		0x0F 0x6F 0x06						/* movq	mm0, [esi] 																*/\
		0x0F 0x6F 0x4E 0x08				/* movq	mm1, [esi + 8] 														*/\
		0x0F 0x6F 0x56 0x10				/* movq	mm2, [esi + 16] 														*/\
		0x0F 0x6F 0x5E 0x18				/* movq	mm3, [esi + 24] 														*/\
		0x83 0xC6 0x20                /* add	esi, 32	;move the data pointer ahead 32						*/\
		0x0F 0xFC 0xE0						/* paddb	mm4, mm0 																*/\
		0x0F 0xEF 0xE8						/* pxor	mm5, mm0 																*/\
		0x0F 0xFC 0xE1						/* paddb	mm4, mm1 																*/\
		0x0F 0xEF 0xE9						/* pxor	mm5, mm1 																*/\
		0x0F 0xFC 0xE2						/* paddb	mm4, mm2 																*/\
		0x0F 0xEF 0xEA						/* pxor	mm5, mm2 																*/\
		0x0F 0xFC 0xE3						/* paddb	mm4, mm3 																*/\
		0x0F 0xEF 0xEB						/* pxor	mm5, mm3 																*/\
		0x49                          /* dec	ecx		;see if there is more to do							*/\
		0x75 0xD3                     /* jnz	#BigStuffLoop 															*/\
		0x0F 0x7E 0xEB						/* movd	ebx, mm5 																*/\
		0x0F 0x73 0xD5 0x20				/* psrlq	mm5, 32 																	*/\
		0x0F 0x7E 0xE8						/* movd	eax, mm5 																*/\
		0x33 0xD8                     /* xor	ebx, eax 																*/\
		0x0F 0x6F 0xC4						/* movq	mm0, mm4 																*/\
		0x0F 0x73 0xD0 0x20				/* psrlq	mm0, 32 																	*/\
		0x0F 0xFC 0xE0						/* paddb	mm4, mm0 																*/\
		0x0F 0x6F 0xC4						/* movq	mm0, mm4 																*/\
		0x0F 0x73 0xD0 0x10				/* psrlq	mm0, 16 																	*/\
		0x0F 0xFC 0xE0						/* paddb	mm4, mm0 																*/\
		0x0F 0x7E 0xE2						/* movd	edx, mm4 																*/\
		0x0F 0x77							/* emms	;end of MMX stuff 													*/\
		0x8B 0xCF                     /* mov	ecx, edi	;load up the rest of the length						*/\
		0x83 0xF9 0x04                /* cmp	ecx, 4 																	*/\
		0x0F 0x82 0x1D 0x00 0x00 0x00	/* jb		#SmallStuff 															*/\
		0xC1 0xE9 0x02                /* shr	ecx, 2 																	*/\
		0x83 0xE7 0x03                /* and	edi, 3 																	*/\
		0x8B 0x06                     /* mov	eax, [esi] 																*/\
		0x83 0xC6 0x04                /* add	esi, 4 																	*/\
		0x33 0xD8                     /* xor	ebx, eax 																*/\
		0x02 0xD0                     /* add	dl, al 																	*/\
		0x02 0xF4                     /* add	dh, ah 																	*/\
		0xC1 0xE8 0x10                /* shr	eax, 16 																	*/\
		0x02 0xD0                     /* add	dl, al 																	*/\
		0x02 0xF4                     /* add	dh, ah 																	*/\
		0x49                          /* dec	ecx 																		*/\
		0x75 0xEB                     /* jnz	#DSSumLoop 																*/\
		0x8B 0xCF                     /* mov	ecx, edi	;load up the rest of the length						*/\
		0x02 0xD6                     /* add	dl, dh	;get complete sum in dl									*/\
		0x8B 0xC3                     /* mov	eax, ebx	;get complete xor in bl									*/\
		0xC1 0xE8 0x10                /* shr	eax, 16 																	*/\
		0x66 0x33 0xD8                /* xor	bx, ax 																	*/\
		0x32 0xDF                     /* xor	bl, bh 																	*/\
		0x83 0xF9 0x00                /* cmp	ecx, 0	;see if anything left to do - 3 or less bytes	*/\
		0x0F 0x84 0x0A 0x00 0x00 0x00	/* jz		#Done 																	*/\
		0x8A 0x06                     /* mov	al, [esi] 																*/\
		0x46                          /* inc	esi																		*/\
		0x02 0xD0                     /* add	dl, al 																	*/\
		0x32 0xD8                     /* xor	bl, al 																	*/\
		0x49                          /* dec	ecx 																		*/\
		0x75 0xF6                     /* jnz	#SmallStuffLoop 														*/\
		0x81 0xE2 0xFF 0x00 0x00 0x00 /* and	edx, 0ffh	;clear unneeded bits									*/\
		0x58                          /* pop	eax 																		*/\
		0x81 0xE3 0xFF 0x00 0x00 0x00 /* and	ebx, 0ffh	;clear unneeded bits									*/\
		0x5F                          /* pop	edi 																		*/\
		0x89 0x18                     /* mov	[eax], ebx 																*/\
		0x89 0x17                     /* mov	[edi], edx 																*/\
		parm [ESI] [eax] [ebx] [ecx]	\
		modify exact [eax ebx ecx edx ESI EDI];
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_32BIT) && defined( FLM_WIN)
static void ftkFastCheckSumMMX(
		void *				pBlk,
		unsigned long *	puiChecksum,	
		unsigned long *	puiXOR,
		unsigned long		uiNumberOfBytes)
{
	__asm
	{
			mov		esi, pBlk

			// Load up the starting checksum values into edx (add) and ebx (XOR)

			mov		eax, puiChecksum
			mov		edx, [eax]
			and		edx, 0ffh			;clear unneeded bits 
			mov		eax, puiXOR
			mov		ebx, [eax]
			and		ebx, 0ffh			;clear unneeded bits 
			mov		ecx, uiNumberOfBytes
			mov		edi, ecx				;save the amount to copy 

			cmp		ecx, 32				;see if we have enough for the big loop 
			jb			MediumStuff 					

			shr		ecx, 5				;convert length to 32 byte blocks
			and		edi, 01fh			;change saved length to remainder
			pxor		mm5, mm5				;wasted space to 16 byte align the upcoming loop - check tHIS..
			
			movd		mm4, edx				;set ADD
			movd		mm5, ebx				;set XOR

BigStuffLoop:
												;load up mm0 - mm3 with 8 bytes each of data.
			movq		mm0, [esi]
			movq		mm1, [esi + 8]
			movq		mm2, [esi + 16]
			movq		mm3, [esi + 24]
			add		esi, 32				;move the data pointer ahead 32
												;add mm0 - mm3 to mm4
												;xor mm0 - mm3 with mm5
			paddb		mm4, mm0
			pxor		mm5, mm0
			paddb		mm4, mm1
			pxor		mm5, mm1
			paddb		mm4, mm2
			pxor		mm5, mm2
			paddb		mm4, mm3
			pxor		mm5, mm3
			dec		ecx					;see if there is more to do
			jnz		BigStuffLoop 
												;mm4 contains the sum to this point
												;mm5 contains the xor to this point
												;edi contains the bytes left 
												;esi points to data left to do
												;extract the xor value from mm5 and put it in ebx
			movd		ebx, mm5
			psrlq		mm5, 32 
			movd		eax, mm5
			xor		ebx, eax
												;extract the sum value from mm4 and put it in dl & dh 
			movq		mm0, mm4
			psrlq		mm0, 32 
			paddb		mm4, mm0
			movq		mm0, mm4
			psrlq		mm0, 16 
			paddb		mm4, mm0
			movd		edx, mm4
			emms								;end of MMX stuff

			mov		ecx, edi				;load up the rest of the length 
												;dl contains half the sum to this point
												;dh contains half the sum to this point
												;ebx contains the xor to this point - 32 bits wide.
												;ecx contains the bytes still left to do 
												;esi contains pointer to data to checksum 
MediumStuff:
			cmp		ecx, 4
			jb			SmallStuff
			shr		ecx, 2
			and		edi, 3

DSSumLoop:
			mov		eax, [esi]
			add		esi, 4
			xor		ebx, eax
			add		dl, al
			add		dh, ah
			shr		eax, 16
			add		dl, al
			add		dh, ah
			dec		ecx
			jnz		DSSumLoop
			mov		ecx, edi				;load up the rest of the length 
												;dl contains half the sum to this point 
												;dh contains half the sum to this point 
												;ebx contains the xor to this point - 32 bits wide.
												;ecx contains the bytes still left to do 
												;esi contains pointer to data to checksum 
SmallStuff:
			add		dl, dh				;get complete sum in dl 
			mov		eax, ebx				;get complete xor in bl
			shr		eax, 16 						
			xor		bx, ax 							
			xor		bl, bh 							
			cmp		ecx, 0				;see if anything left to do - 3 or less bytes 
			jz			Done 							

SmallStuffLoop: 						
			mov		al, [esi] 						
			inc		esi								
			add		dl, al 							
			xor		bl, al 							
			dec		ecx 							
			jnz		SmallStuffLoop 				
Done: 									
			and		edx, 0ffh			;clear unneeded bits 
			and		ebx, 0ffh			;clear unneeded bits 

			// Set the return values.

			mov		eax, puiChecksum
			mov		[eax], edx

			mov		eax, puiXOR
			mov		[eax], ebx
	}
	return;
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_32BIT) && defined( FLM_GNUC)
static void ftkFastCheckSumMMX(
		void *				pBlk,
		unsigned long *	puiChecksum,	
		unsigned long *	puiXOR,
		unsigned long		uiNumberOfBytes)
{
	__asm__ __volatile__(
			"			push		%%ebx\n"
			"			mov		%2, %%esi\n"
			"			mov		%3, %%eax\n"
			"			mov		(%%eax), %%edx\n"
			"			and		$0xFF, %%edx\n"
			"			mov		%4, %%eax\n"
			"			mov		(%%eax), %%ebx\n"
			"			and		$0xFF, %%ebx\n" 
			"			mov		%5, %%ecx\n"
			"			mov		%%ecx, %%edi\n" 
			
			"			cmp		$32, %%ecx\n"
			"			jb			MediumStuff\n" 					
			
			"			shr		$5, %%ecx\n"
			"			and		$0x01F, %%edi\n"
			"			pxor		%%mm5, %%mm5\n"
						
			"			movd		%%edx, %%mm4\n"
			"			movd		%%ebx, %%mm5\n"
			
			"BigStuffLoop:\n"
			"			movq		(%%esi), %%mm0\n"
			"			movq		8(%%esi), %%mm1\n"
			"			movq		16(%%esi), %%mm2\n"
			"			movq		24(%%esi), %%mm3\n"
			"			add		$32, %%esi\n"
			"			paddb		%%mm0, %%mm4\n"
			"			pxor		%%mm0, %%mm5\n"
			"			paddb		%%mm1, %%mm4\n"
			"			pxor		%%mm1, %%mm5\n"
			"			paddb		%%mm2, %%mm4\n"
			"			pxor		%%mm2, %%mm5\n"
			"			paddb		%%mm3, %%mm4\n"
			"			pxor		%%mm3, %%mm5\n"
			"			dec		%%ecx\n"
			
			"			jnz		BigStuffLoop\n" 
			"			movd		%%mm5, %%ebx\n"
			"			psrlq		$32, %%mm5\n"
			"			movd		%%mm5, %%eax\n"
			"			xor		%%eax, %%ebx\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$32, %%mm0\n"
			"			paddb		%%mm0, %%mm4\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$16, %%mm0\n" 
			"			paddb		%%mm0, %%mm4\n"
			"			movd		%%mm4, %%edx\n"
			"			emms\n"
			
			"			mov		%%edi, %%ecx\n"
			
			"MediumStuff:\n"
			"			cmp		$4, %%ecx\n"
			"			jb			SmallStuff\n"
			"			shr		$2, %%ecx\n"
			"			and		$3, %%edi\n"
			
			"DSSumLoop:\n"
			"			mov		(%%esi), %%eax\n"
			"			add		$4, %%esi\n"
			"			xor		%%eax, %%ebx\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			shr		$16, %%eax\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			dec		%%ecx\n"
			"			jnz		DSSumLoop\n"
			"			mov		%%edi, %%ecx\n"
			
			"SmallStuff:\n"
			"			add		%%dh, %%dl\n" 
			"			mov		%%ebx, %%eax\n"
			"			shr		$16, %%eax\n"				
			"			xor		%%ax, %%bx\n" 							
			"			xor		%%bh, %%bl\n"							
			"			cmp		$0, %%ecx\n" 
			"			jz			Done\n" 							
			
			"SmallStuffLoop:\n" 						
			"			mov		(%%esi), %%al\n" 						
			"			inc		%%esi\n"								
			"			add		%%al, %%dl\n" 							
			"			xor		%%al, %%bl\n" 							
			"			dec		%%ecx\n" 							
			"			jnz		SmallStuffLoop\n" 				
			"Done:\n" 									
			"			and		$0xFF, %%edx\n" 
			"			and		$0xFF, %%ebx\n" 
			
			"			mov		%0, %%eax\n"
			"			mov		%%edx, (%%eax)\n"
			
			"			mov		%1, %%eax\n"
			"			mov		%%ebx, (%%eax)\n"
			"			pop		%%ebx\n"
				: "=m" (puiChecksum), "=m" (puiXOR)
				: "m" (pBlk), "m" (puiChecksum), "m" (puiXOR), "m" (uiNumberOfBytes)
				: "%eax", "%ecx", "%edx", "%esi", "%edi");
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_64BIT) && defined( FLM_GNUC)
static void ftkFastCheckSumMMX(
		void *				pBlk,
		unsigned long *	puiChecksum,	
		unsigned long *	puiXOR,
		unsigned long		uiNumberOfBytes)
{
	__asm__ __volatile__(
			"			mov		%2, %%r8\n"
			"			mov		%3, %%r9\n"
			"			mov		(%%r9), %%edx\n"
			"			and		$0xFF, %%edx\n"
			"			mov		%4, %%r9\n"
			"			mov		(%%r9), %%ebx\n"
			"			and		$0xFF, %%ebx\n" 
			"			mov		%5, %%ecx\n"
			"			mov		%%ecx, %%edi\n" 
			
			"			cmp		$32, %%ecx\n"
			"			jb			MediumStuff\n" 					
			
			"			shr		$5, %%ecx\n"
			"			and		$0x01F, %%edi\n"
			"			pxor		%%mm5, %%mm5\n"
						
			"			movd		%%edx, %%mm4\n"
			"			movd		%%ebx, %%mm5\n"
			
			"BigStuffLoop:\n"
			"			movq		(%%r8), %%mm0\n"
			"			movq		8(%%r8), %%mm1\n"
			"			movq		16(%%r8), %%mm2\n"
			"			movq		24(%%r8), %%mm3\n"
			"			add		$32, %%r8\n"
			"			paddb		%%mm0, %%mm4\n"
			"			pxor		%%mm0, %%mm5\n"
			"			paddb		%%mm1, %%mm4\n"
			"			pxor		%%mm1, %%mm5\n"
			"			paddb		%%mm2, %%mm4\n"
			"			pxor		%%mm2, %%mm5\n"
			"			paddb		%%mm3, %%mm4\n"
			"			pxor		%%mm3, %%mm5\n"
			"			dec		%%ecx\n"
			
			"			jnz		BigStuffLoop\n" 
			"			movd		%%mm5, %%ebx\n"
			"			psrlq		$32, %%mm5\n"
			"			movd		%%mm5, %%eax\n"
			"			xor		%%eax, %%ebx\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$32, %%mm0\n"
			"			paddb		%%mm0, %%mm4\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$16, %%mm0\n" 
			"			paddb		%%mm0, %%mm4\n"
			"			movd		%%mm4, %%edx\n"
			"			emms\n"
			
			"			mov		%%edi, %%ecx\n"
			
			"MediumStuff:\n"
			"			cmp		$4, %%ecx\n"
			"			jb			SmallStuff\n"
			"			shr		$2, %%ecx\n"
			"			and		$3, %%edi\n"
			
			"DSSumLoop:\n"
			"			mov		(%%r8), %%eax\n"
			"			add		$4, %%r8\n"
			"			xor		%%eax, %%ebx\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			shr		$16, %%eax\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			dec		%%ecx\n"
			"			jnz		DSSumLoop\n"
			"			mov		%%edi, %%ecx\n"
			
			"SmallStuff:\n"
			"			add		%%dh, %%dl\n" 
			"			mov		%%ebx, %%eax\n"
			"			shr		$16, %%eax\n"				
			"			xor		%%ax, %%bx\n" 							
			"			xor		%%bh, %%bl\n"							
			"			cmp		$0, %%ecx\n" 
			"			jz			Done\n" 							
			
			"SmallStuffLoop:\n" 						
			"			mov		(%%r8), %%al\n" 						
			"			inc		%%r8\n"								
			"			add		%%al, %%dl\n"
			"			xor		%%al, %%bl\n" 							
			"			dec		%%ecx\n" 							
			"			jnz		SmallStuffLoop\n" 				
			"Done:\n" 									
			"			and		$0xFF, %%edx\n" 
			"			and		$0xFF, %%ebx\n" 
			
			"			mov		%0, %%r9\n"
			"			mov		%%edx, (%%r9)\n"
			
			"			mov		%1, %%r9\n"
			"			mov		%%ebx, (%%r9)\n"
				: "=m" (puiChecksum), "=m" (puiXOR)
				: "m" (pBlk), "m" (puiChecksum), "m" (puiXOR), "m" (uiNumberOfBytes)
				: "%eax", "%ebx", "%ecx", "%edi", "%edx", "%r8", "%r9");
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SPARC_PLUS)
void sparc_csum_code( void)
{
	// Calculate the sum and xor bytes of a checksum

	asm( ".global sparc_calc_checksum");
#ifdef FLM_64BIT
	asm( ".align 8");
#else
	asm( ".align 4");
#endif
	asm( "sparc_calc_checksum:");
	asm( "	save %sp, -96, %sp");

#ifdef FLM_64BIT
	asm( "	ldx [%i1], %l1");
	asm( "	ldx [%i2], %l2");
#else
	asm( "	ld [%i1], %l1");
	asm( "	ld [%i2], %l2");
#endif

	asm( "	mov %i0, %l0");
	asm( "	mov %i3, %l3");

	asm( "	clr %l4");

	asm( "	csum_loop:");
	asm( "		ldub [%l0], %l4");
	asm( "		add %l4, %l1, %l1");
	asm( "		xor %l4, %l2, %l2");
	asm( "		inc %l0");
	asm( "		subcc %l3, 1, %l3");
	asm( "		bg csum_loop");
	asm( "		nop");

	asm( "	and %l1, 0xFF, %l1");
	asm( "	and %l2, 0xFF, %l2");
#ifdef FLM_64BIT
	asm( "	stx %l1, [%i1]");
	asm( "	stx %l2, [%i2]");
#else
	asm( "	st %l1, [%i1]");
	asm( "	st %l2, [%i2]");
#endif
	asm( "	ret");
	asm( "	restore");
	asm( ".type sparc_calc_checksum, #function");
	asm( ".size sparc_calc_checksum, (.-sparc_calc_checksum)");

	// Calculate the xor byte of a checksum

	asm( ".global sparc_calc_xor");
#ifdef FLM_64BIT
	asm( ".align 8");
#else
	asm( ".align 4");
#endif
	asm( "sparc_calc_xor:");
	asm( "	save %sp, -96, %sp");

	asm( "	mov %i0, %l0");
#ifdef FLM_64BIT
	asm( "	ldx [%i1], %l1");
#else
	asm( "	ld [%i1], %l1");
#endif

	asm( "	clr %l3");

	asm( "	mov %i2, %l2");
#ifdef FLM_64BIT
	asm( "	and %l2, 0x7, %l2");
#else
	asm( "	and %l2, 0x3, %l2");
#endif
	asm( "	cmp %l2, 0");
	asm( "	be init_main_xor_loop");
	asm( "	nop");

	asm( "	lead_xor_loop:");
#ifdef FLM_64BIT
	asm( "		ldx [%l0], %l3");
#else
	asm( "		ld [%l0], %l3");
#endif
	asm( "		xor %l3, %l1, %l1");
	asm( "		inc %l0");
	asm( "		deccc %l2");
	asm( "		bg lead_xor_loop");
	asm( "		nop");

	asm( "	init_main_xor_loop:");
	asm( "		mov %i2, %l2");
#ifdef FLM_64BIT
	asm( "		andn %l2, 0x7, %l2");
	asm( "		cmp %l2, 8");
#else
	asm( "		andn %l2, 0x3, %l2");
	asm( "		cmp %l2, 4");
#endif
	asm( "		bl init_tail_xor_loop");
	asm( "		nop");

	asm( "	main_xor_loop:");
#ifdef FLM_64BIT
	asm( "		ldx [%l0], %l3");
#else
	asm( "		ld [%l0], %l3");
#endif
	asm( "		xor %l3, %l1, %l1");
#ifdef FLM_64BIT
	asm( "		add %l0, 8, %l0");
	asm( "		subcc %l2, 8, %l2");
#else
	asm( "		add %l0, 4, %l0");
	asm( "		subcc %l2, 4, %l2");
#endif
	asm( "		bg main_xor_loop");
	asm( "		nop");

	asm( "	init_tail_xor_loop:");
	asm( "		cmp %i2, 0");
	asm( "		be done");
	
	asm( "	tail_xor_loop:");
#ifdef FLM_64BIT
	asm( "		ldx [%l0], %l3");
#else
	asm( "		ld [%l0], %l3");
#endif
	asm( "		xor %l3, %l1, %l1");
	asm( "		inc %l0");
	asm( "		deccc %l2");
	asm( "		bg tail_xor_loop");
	asm( "		nop");

	asm( "	done:");
	asm( "		mov %l1, %l3");
#ifdef FLM_64BIT
	asm( "		mov 7, %l2");
#else
	asm( "		mov 3, %l2");
#endif
	asm( "		xor_assemble_loop:");
	asm( "			srlx %l3, 8, %l3");
	asm( "			xor %l3, %l1, %l1");
	asm( "			subcc %l2, 1, %l2");
	asm( "			bg xor_assemble_loop");
	asm( "			nop");
	asm( "		and %l1, 0xFF, %l1");
#ifdef FLM_64BIT
	asm( "		stx %l1, [%i1]");
#else
	asm( "		st %l1, [%i1]");
#endif
	asm( "	ret");
	asm( "	restore");
	asm( ".type sparc_calc_xor, #function");
	asm( ".size sparc_calc_xor, (.-sparc_calc_xor)");
}
#endif

/******************************************************************************
Desc: Sets the global variable to check if MMX instructions are allowed.
******************************************************************************/
void f_initFastCheckSum( void)
{
#if defined( FLM_X86) && (defined( FLM_GNUC) || defined( FLM_WIN) || defined( FLM_NLM))
	// NOTE that ftkGetMMXSupported assumes that we are running on at least a
	// pentium.  The check to see if we are on a pentium requires that  we
	// modify the flags register, and we can't do that if we are running
	// in ring3.  Because NetWare 5 - according to our product marketing -
	// requires at least a P5 90Mhz, we will be safe.  When you port this
	// code to NT, you may need to come up with a safe way to see if we
	// can do MMX instructions - unless you can assume that even on NT you
	// will be on at least a P5.

	gv_bHaveFastCheckSum = ftkGetMMXSupported() ? TRUE : FALSE;
#elif defined( FLM_SPARC_PLUS)
	gv_bHaveFastCheckSum = TRUE;
#endif
}

/********************************************************************
Desc:	Calculate the checksum for a block.  NOTE: This is ALWAYS done
		on the raw image that will be written to disk.  This means
		that if the block needs to be converted before writing it out,
		it should be done before calculating the checksum.
*********************************************************************/
FLMUINT32 FLMAPI f_calcFastChecksum(
	const void *	pvData,
	FLMUINT			uiLength,
	FLMUINT *		puiAdds,
	FLMUINT *		puiXORs)
{
	FLMUINT			uiAdds = 0;
	FLMUINT			uiXORs = 0;
	FLMBYTE *		pucData = (FLMBYTE *)pvData;
	
	if( puiAdds)
	{
		uiAdds = *puiAdds;
	}
	
	if( puiXORs)
	{
		uiXORs = *puiXORs;
	}

#if defined( FLM_X86) && (defined( FLM_GNUC) || defined( FLM_WIN) || defined( FLM_NLM))
	if( gv_bHaveFastCheckSum)
	{
		ftkFastCheckSumMMX( (void *) pucData, (unsigned long *) &uiAdds, 
					(unsigned long *) &uiXORs, (unsigned long) uiLength);
	}
#elif defined( FLM_SPARC_PLUS)
	sparc_calc_checksum( pucData, &uiAdds, &uiXORs, uiLength);
#endif

	if( !gv_bHaveFastCheckSum)
	{
		FLMBYTE *		pucCur = pucData;
		FLMBYTE *		pucEnd = pucData + uiLength;
	
		while( pucCur < pucEnd)	
		{
			uiAdds += *pucCur;
			uiXORs ^= *pucCur++;
		}
	
		uiAdds &= 0xFF;
	}

	if( puiAdds)
	{
		*puiAdds = uiAdds;
	}
	
	if( puiXORs)
	{
		*puiXORs = uiXORs;
	}
	
	return( (FLMUINT32)((uiAdds << 16) + uiXORs));
}

/****************************************************************************
Desc: Generates a table of remainders for each 8-bit byte.  The resulting
		table is used by f_updateCRC to calculate a CRC value.  The table
		must be freed via a call to f_free.
*****************************************************************************/
RCODE f_initCRCTable( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT32 *		pTable;
	FLMUINT32		ui32Val;
	FLMUINT32		ui32Loop;
	FLMUINT32		ui32SubLoop;

	// Use the standard degree-32 polynomial used by
	// Ethernet, PKZIP, etc. for computing the CRC of
	// a data stream.  This is the little-endian
	// representation of the polynomial.  The big-endian
	// representation is 0x04C11DB7.

#define CRC_POLYNOMIAL		((FLMUINT32)0xEDB88320)

	f_assert( !gv_pui32CRCTbl);

	if( RC_BAD( rc = f_alloc( 256 * sizeof( FLMUINT32), &pTable)))
	{
		goto Exit;
	}

	for( ui32Loop = 0; ui32Loop < 256; ui32Loop++)
	{
		ui32Val = ui32Loop;
		for( ui32SubLoop = 0; ui32SubLoop < 8; ui32SubLoop++)
		{
			if( ui32Val & 0x00000001)
			{
				ui32Val = CRC_POLYNOMIAL ^ (ui32Val >> 1);
			}
			else
			{
				ui32Val >>= 1;
			}
		}

		pTable[ ui32Loop] = ui32Val;
	}

	gv_pui32CRCTbl = pTable;
	pTable = NULL;

Exit:

	if( pTable)
	{
		f_free( &pTable);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void f_freeCRCTable( void)
{
	if( gv_pui32CRCTbl)
	{
		f_free( &gv_pui32CRCTbl);
	}
}
	
/****************************************************************************
Desc: Computes the CRC of the passed-in data buffer.  Multiple calls can
		be made to this routine to build a CRC over multiple data buffers.
		On the first call, *pui32CRC must be initialized to something
		(0, etc.).  For generating CRCs that are compatible with PKZIP,
		*pui32CRC should be initialized to 0xFFFFFFFF and the ones complement
		of the resulting CRC should be computed.
*****************************************************************************/
void FLMAPI f_updateCRC(
	const void *		pvBuffer,
	FLMUINT				uiCount,
	FLMUINT32 *			pui32CRC)
{
	FLMBYTE *			pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT32			ui32CRC = *pui32CRC;
	FLMUINT				uiLoop;

	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		ui32CRC = (ui32CRC >> 8) ^ gv_pui32CRCTbl[
			((FLMBYTE)(ui32CRC & 0x000000FF)) ^ pucBuffer[ uiLoop]];
	}

	*pui32CRC = ui32CRC;
}

/********************************************************************
Desc:
*********************************************************************/
FLMBYTE FLMAPI f_calcPacketChecksum(
	FLMBYTE *		pucPacket,
	FLMUINT			uiBytesToChecksum)
{
	FLMUINT			uiChecksum = 0;
	unsigned long	uiAdds = 0;
	unsigned long	uiXORs = 0;
	
#if defined( FLM_X86) && (defined( FLM_GNUC) || defined( FLM_WIN) || defined( FLM_NLM))
	if( gv_bHaveFastCheckSum)
	{
		ftkFastCheckSumMMX( pucPacket, &uiAdds, &uiXORs, uiBytesToChecksum);
		uiChecksum = uiXORs;
	}
#elif defined( FLM_SPARC_PLUS)
	sparc_calc_xor( pucPacket, &uiXORs, uiBytesToChecksum);
	uiChecksum = uiXORs;
#endif

	if( !gv_bHaveFastCheckSum)
	{
		FLMBYTE *	pucEnd;
		FLMBYTE *	pucSectionEnd;
		FLMBYTE *	pucCur;
		FLMBYTE		ucTmp;
		
		pucCur = pucPacket;
		pucEnd = pucPacket + uiBytesToChecksum;
	
	#ifdef FLM_64BIT
		pucSectionEnd = pucPacket + (sizeof( FLMUINT) - ((FLMUINT)pucPacket & 0x7));
	#else
		pucSectionEnd = pucPacket + (sizeof( FLMUINT) - ((FLMUINT)pucPacket & 0x3));
	#endif
	
		if( pucSectionEnd > pucEnd)
		{
			pucSectionEnd = pucEnd;
		}
	
		while( pucCur < pucSectionEnd)
		{
			uiChecksum = (uiChecksum << 8) + *pucCur++;
		}
	
	#ifdef FLM_64BIT
		pucSectionEnd = (FLMBYTE *)((FLMUINT)pucEnd & 0xFFFFFFFFFFFFFFF8); 
	#else
		pucSectionEnd = (FLMBYTE *)((FLMUINT)pucEnd & 0xFFFFFFFC); 
	#endif
	
		while( pucCur < pucSectionEnd)
		{
			uiChecksum ^= *((FLMUINT *)pucCur);
			pucCur += sizeof( FLMUINT);
		}
	
		while( pucCur < pucEnd)
		{
			uiChecksum ^= *pucCur++;
		}
	
		ucTmp = (FLMBYTE)uiChecksum;
	
		uiChecksum >>= 8;
		ucTmp ^= (FLMBYTE)uiChecksum;
	
		uiChecksum >>= 8;
		ucTmp ^= (FLMBYTE)uiChecksum;
	
	#ifdef FLM_64BIT
		uiChecksum >>= 8;
		ucTmp ^= (FLMBYTE)uiChecksum;
	
		uiChecksum >>= 8;
		ucTmp ^= (FLMBYTE)uiChecksum;
	
		uiChecksum >>= 8;
		ucTmp ^= (FLMBYTE)uiChecksum;
	
		uiChecksum >>= 8;
		ucTmp ^= (FLMBYTE)uiChecksum;
	#endif
	
		ucTmp ^= (FLMBYTE)(uiChecksum >> 8);
		uiChecksum = (FLMUINT)ucTmp;
	}

	return( (FLMBYTE)(uiChecksum != 0 ? uiChecksum : 1));
}
