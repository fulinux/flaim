//------------------------------------------------------------------------------
// Desc:
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: $
//------------------------------------------------------------------------------

#include "xflaim.h"
#include "jniftk.h"

/****************************************************************************
Desc:
****************************************************************************/
void ThrowError(
	RCODE				rc,
	JNIEnv *			pEnv)
{
  char 				szMsg[ 128];
  
  f_strcpy( szMsg, "Bad RCODE: ");
  
  jclass class_XFlaimException = pEnv->FindClass( "xflaim/XFlaimException");
  jmethodID id_Constructor = pEnv->GetMethodID( class_XFlaimException,
  												"<init>",
  												"(ILjava/lang/String;)V");

  jobject Exception = pEnv->NewObject( class_XFlaimException, id_Constructor,
  									   (jint)rc, pEnv->NewStringUTF( szMsg));

  pEnv->Throw( reinterpret_cast<jthrowable>(Exception));
}
